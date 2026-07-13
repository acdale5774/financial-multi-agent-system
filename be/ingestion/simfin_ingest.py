"""Ingest SimFin structured financial data into Postgres.

Pipeline:
    fetch_simfin_data()  ->  normalize_simfin_data()  ->  load_into_postgres()

Access method: the official `simfin` SDK, which downloads whole bulk datasets
(income / balance / cashflow) as pandas DataFrames and caches them on disk.
Data is normalized into a **long/tall** shape — one row per
(company, statement, fiscal period, metric) — so a single `financials` table
covers all statement types and any metric set.

Run it:
    python -m ingestion.simfin_ingest --init-db --variant annual
    python -m ingestion.simfin_ingest --variant quarterly --statements income,balance,cashflow
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import pandas as pd

# Statements we know how to ingest, mapped to their SimFin loader name.
STATEMENTS: tuple[str, ...] = ("income", "balance", "cashflow")

# DataFrame columns that describe a row rather than being a financial metric.
# Everything else in a SimFin statement DataFrame is treated as a metric.
_META_COLUMNS: frozenset[str] = frozenset(
    {
        "index",
        "Ticker",
        "SimFinId",
        "Company Name",
        "Currency",
        "Fiscal Year",
        "Fiscal Period",
        "Report Date",
        "Publish Date",
        "Restated Date",
        "Source",
    }
)


@dataclass(frozen=True)
class FinancialRecord:
    """One normalized financial data point, ready to load into `financials`."""

    ticker: str
    simfin_id: str | None
    company_name: str | None
    statement: str
    fiscal_year: int
    fiscal_period: str  # 'FY' for annual; 'Q1'..'Q4' for quarterly
    currency: str | None
    report_date: date | None
    metric: str  # snake_cased, e.g. 'revenue', 'net_income'
    value: float


# --------------------------------------------------------------------------
# 1. Fetch
# --------------------------------------------------------------------------


def _configure_simfin(api_key: str | None = None, data_dir: str | None = None) -> None:
    """Point the SimFin SDK at the API key and local cache directory."""
    import simfin as sf  # imported lazily so normalize/tests don't require it

    from core.config import settings

    sf.set_api_key(api_key or settings.simfin_api_key)
    sf.set_data_dir(data_dir or settings.simfin_data_dir)


def fetch_simfin_data(
    statement: str,
    *,
    market: str = "us",
    variant: str = "annual",
    api_key: str | None = None,
    data_dir: str | None = None,
) -> pd.DataFrame:
    """Fetch a raw SimFin statement dataset as a pandas DataFrame.

    Args:
        statement: One of "income", "balance", "cashflow".
        market: Market identifier (default "us").
        variant: "annual", "quarterly", or "ttm".
        api_key: SimFin API key; falls back to SIMFIN_API_KEY / "free".
        data_dir: Local cache directory; falls back to SIMFIN_DATA_DIR.

    Returns:
        The raw dataset (wide: one row per company/period, one column per metric).
    """
    import simfin as sf

    _configure_simfin(api_key, data_dir)
    loaders = {
        "income": sf.load_income,
        "balance": sf.load_balance,
        "cashflow": sf.load_cashflow,
    }
    if statement not in loaders:
        raise ValueError(
            f"Unknown statement '{statement}'. Expected one of {sorted(loaders)}."
        )
    return loaders[statement](variant=variant, market=market)


def _load_company_info(
    *,
    market: str,
    api_key: str | None = None,
    data_dir: str | None = None,
) -> dict[str, dict[str, str | None]]:
    """Load the SimFin companies dataset into a ticker -> {name, simfin_id, sector} map."""
    import simfin as sf

    _configure_simfin(api_key, data_dir)
    try:
        companies = sf.load_companies(market=market)
    except Exception as exc:  # noqa: BLE001 - metadata is best-effort
        print(f"warning: could not load company metadata ({exc}); using tickers as names.")
        return {}

    info: dict[str, dict[str, str | None]] = {}
    for row in _flatten(companies).to_dict("records"):
        ticker = _opt_str(row.get("Ticker"))
        if not ticker:
            continue
        info[ticker] = {
            "simfin_id": _opt_str(row.get("SimFinId")),
            "name": _opt_str(row.get("Company Name")) or ticker,
            # Sector name may require the industries dataset; left as best-effort.
            # TODO: enrich sector via sf.load_industries() join on IndustryId.
            "sector": _opt_str(row.get("Sector")),
        }
    return info


# --------------------------------------------------------------------------
# 2. Normalize
# --------------------------------------------------------------------------


def normalize_simfin_data(raw: pd.DataFrame, *, statement: str) -> list[FinancialRecord]:
    """Normalize a wide SimFin statement DataFrame into long/tall records.

    Melts every non-metadata column into (metric, value) pairs, tags each with
    its statement + fiscal period, and skips null values. Metric names are
    snake_cased (e.g. "Net Income" -> "net_income").

    Args:
        raw: A DataFrame from `fetch_simfin_data`.
        statement: The statement these rows belong to ("income"/"balance"/"cashflow").

    Returns:
        Normalized `FinancialRecord`s aligned to the `financials` table.
    """
    df = _flatten(raw)
    metric_cols = [c for c in df.columns if c not in _META_COLUMNS]

    records: list[FinancialRecord] = []
    for row in df.to_dict("records"):
        ticker = _opt_str(row.get("Ticker"))
        fiscal_year = row.get("Fiscal Year")
        fiscal_period = _opt_str(row.get("Fiscal Period"))
        if not ticker or fiscal_period is None or pd.isna(fiscal_year):
            continue

        simfin_id = _opt_str(row.get("SimFinId"))
        company_name = _opt_str(row.get("Company Name"))
        currency = _opt_str(row.get("Currency"))
        report_date = _opt_date(row.get("Report Date"))

        for col in metric_cols:
            raw_value = row.get(col)
            if raw_value is None or pd.isna(raw_value):
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue  # non-numeric stray column; not a metric
            records.append(
                FinancialRecord(
                    ticker=ticker,
                    simfin_id=simfin_id,
                    company_name=company_name,
                    statement=statement,
                    fiscal_year=int(fiscal_year),
                    fiscal_period=fiscal_period,
                    currency=currency,
                    report_date=report_date,
                    metric=_metric_key(col),
                    value=value,
                )
            )
    return records


# --------------------------------------------------------------------------
# 3. Load
# --------------------------------------------------------------------------


def load_into_postgres(
    records: list[FinancialRecord],
    *,
    company_info: dict[str, dict[str, str | None]] | None = None,
    database_url: str | None = None,
) -> int:
    """Upsert normalized records into `companies` and `financials`.

    Companies are upserted first (keyed on ticker) so financial rows can resolve
    their `company_id`. Both writes are idempotent via `ON CONFLICT ... DO UPDATE`,
    so re-running an ingest refreshes values instead of duplicating them.

    Args:
        records: Output of `normalize_simfin_data`.
        company_info: Optional ticker -> {name, simfin_id, sector} enrichment.
        database_url: Override Postgres connection string.

    Returns:
        Number of `financials` rows inserted or updated.
    """
    if not records:
        return 0

    from db.connection import get_connection

    company_info = company_info or {}
    tickers = sorted({r.ticker for r in records})

    # Fallback company attributes taken from the statement rows themselves.
    name_from_records: dict[str, str] = {}
    simfin_from_records: dict[str, str] = {}
    for r in records:
        if r.company_name:
            name_from_records.setdefault(r.ticker, r.company_name)
        if r.simfin_id:
            simfin_from_records.setdefault(r.ticker, r.simfin_id)

    company_rows = [
        (
            ticker,
            company_info.get(ticker, {}).get("name")
            or name_from_records.get(ticker)
            or ticker,
            company_info.get(ticker, {}).get("simfin_id") or simfin_from_records.get(ticker),
            company_info.get(ticker, {}).get("sector"),
        )
        for ticker in tickers
    ]

    with get_connection(database_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO companies (ticker, name, simfin_id, sector)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (ticker) DO UPDATE
                    SET name = EXCLUDED.name,
                        simfin_id = COALESCE(EXCLUDED.simfin_id, companies.simfin_id),
                        sector = COALESCE(EXCLUDED.sector, companies.sector)
                """,
                company_rows,
            )

            cur.execute(
                "SELECT id, ticker FROM companies WHERE ticker = ANY(%s)",
                (tickers,),
            )
            id_by_ticker = {row["ticker"]: row["id"] for row in cur.fetchall()}

            financial_rows = [
                (
                    id_by_ticker[r.ticker],
                    r.statement,
                    r.fiscal_year,
                    r.fiscal_period,
                    r.currency,
                    r.report_date,
                    r.metric,
                    r.value,
                )
                for r in records
            ]
            cur.executemany(
                """
                INSERT INTO financials
                    (company_id, statement, fiscal_year, fiscal_period,
                     currency, report_date, metric, value)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (company_id, statement, fiscal_year, fiscal_period, metric)
                DO UPDATE SET value = EXCLUDED.value,
                              currency = EXCLUDED.currency,
                              report_date = EXCLUDED.report_date
                """,
                financial_rows,
            )
        conn.commit()

    return len(financial_rows)


# --------------------------------------------------------------------------
# Orchestration + CLI
# --------------------------------------------------------------------------


def ingest(
    statements: list[str],
    *,
    market: str | None = None,
    variant: str = "annual",
    api_key: str | None = None,
    data_dir: str | None = None,
    init_db: bool = False,
    database_url: str | None = None,
) -> int:
    """Run the full fetch -> normalize -> load pipeline for each statement.

    Returns the total number of `financials` rows upserted.
    """
    from core.config import settings

    market = market or settings.simfin_market

    if init_db:
        from db.connection import apply_schema

        apply_schema(database_url)

    company_info = _load_company_info(market=market, api_key=api_key, data_dir=data_dir)

    total = 0
    for statement in statements:
        raw = fetch_simfin_data(
            statement, market=market, variant=variant, api_key=api_key, data_dir=data_dir
        )
        records = normalize_simfin_data(raw, statement=statement)
        loaded = load_into_postgres(
            records, company_info=company_info, database_url=database_url
        )
        total += loaded
        print(f"[{statement}] {variant}/{market}: {len(records)} records -> {loaded} rows upserted")
    return total


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Move a named/multi index into columns so meta fields are addressable."""
    if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
        return df.reset_index()
    return df


def _metric_key(name: str) -> str:
    """Snake_case a SimFin column name: 'Shares (Diluted)' -> 'shares_diluted'."""
    key = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", key).strip("_")


def _opt_str(value: object) -> str | None:
    """Coerce a cell to a clean string, or None for missing values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and value.is_integer():
        value = int(value)  # avoid '12345.0' for id-like fields
    text = str(value).strip()
    return text or None


def _opt_date(value: object) -> date | None:
    """Coerce a cell to a date, or None if it can't be parsed."""
    if value is None:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if timestamp is None or pd.isna(timestamp):
        return None
    return timestamp.date()


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest SimFin structured financials into Postgres."
    )
    parser.add_argument(
        "--statements",
        default=",".join(STATEMENTS),
        help="Comma-separated statements to ingest (default: income,balance,cashflow).",
    )
    parser.add_argument(
        "--variant",
        default="annual",
        choices=["annual", "quarterly", "ttm"],
        help="Reporting granularity (default: annual).",
    )
    parser.add_argument("--market", default=None, help="SimFin market (default: settings).")
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Apply db/schema.sql before loading.",
    )
    args = parser.parse_args(argv)

    statements = [s.strip() for s in args.statements.split(",") if s.strip()]
    total = ingest(
        statements, market=args.market, variant=args.variant, init_db=args.init_db
    )
    print(f"Done. {total} financial rows upserted.")


if __name__ == "__main__":
    main()
