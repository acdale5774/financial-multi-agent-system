"""Ingest SimFin structured financial data into Postgres.

Pipeline:
    fetch_simfin_data()  ->  normalize_simfin_data()  ->  load_into_postgres()

Nothing here is implemented yet — these are typed placeholders that define the
shape of the pipeline so the rest of the system can be built against them.
"""

from __future__ import annotations

from typing import Any

# A single normalized financial record, ready to be written to Postgres.
# TODO: Replace this loose alias with a pydantic model (e.g. FinancialRecord)
#       once the target `financials` table schema is finalized in db/schema.sql.
FinancialRecord = dict[str, Any]


def fetch_simfin_data(
    dataset: str,
    *,
    market: str = "us",
    variant: str = "annual",
    api_key: str | None = None,
) -> Any:
    """Fetch a raw dataset from SimFin.

    Args:
        dataset: SimFin dataset name (e.g. "income", "balance", "cashflow").
        market: Market identifier (default "us").
        variant: Reporting variant (e.g. "annual", "quarterly", "ttm").
        api_key: SimFin API key; falls back to the SIMFIN_API_KEY env var.

    Returns:
        The raw dataset as returned by the SimFin client (shape TBD).

    TODO: Use the official SimFin SDK or the bulk CSV/REST API. Handle
          pagination, rate limits, and on-disk caching of raw downloads.
    """
    raise NotImplementedError


def normalize_simfin_data(raw: Any) -> list[FinancialRecord]:
    """Normalize a raw SimFin dataset into flat, typed records.

    Responsibilities:
        - Map SimFin column/field names to our internal schema.
        - Coerce types (dates, decimals) and units (e.g. reported currency).
        - Attach a stable company identifier so records join to `companies`.
        - Drop or flag incomplete rows.

    Args:
        raw: The raw payload returned by `fetch_simfin_data`.

    Returns:
        A list of normalized records aligned to the `financials` table.

    TODO: Define the canonical field mapping and validation rules.
    """
    raise NotImplementedError


def load_into_postgres(
    records: list[FinancialRecord],
    *,
    database_url: str | None = None,
) -> int:
    """Upsert normalized financial records into Postgres.

    Args:
        records: Normalized records from `normalize_simfin_data`.
        database_url: Postgres connection string; falls back to DATABASE_URL.

    Returns:
        The number of rows inserted or updated.

    TODO: Use psycopg with an idempotent upsert (ON CONFLICT ... DO UPDATE)
          keyed on (company_id, fiscal_period, statement). Batch the writes
          and wrap them in a transaction.
    """
    raise NotImplementedError
