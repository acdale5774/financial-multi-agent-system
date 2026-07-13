"""Enrich structured data with the joins the agent layer needs.

Two backfills, both idempotent:

1. `backfill_sectors`: SimFin's companies dataset carries an IndustryId; the
   separate industries dataset maps it to (Sector, Industry). The original
   ingest left `companies.sector` NULL (the statement datasets don't carry it),
   so this fills `sector` and `industry` for every company SimFin classifies.
2. `backfill_document_companies`: `documents.company_id` links the unstructured
   corpus to SimFin companies (the cross-source join key). Octus company names
   ("Ford Motor") rarely equal SimFin names ("FORD MOTOR CO"), so rows are
   matched on normalized name tokens (legal suffixes stripped) and only a
   UNIQUE match is written — ambiguous or unknown names are reported, not
   guessed.

Run it:
    python -m ingestion.enrich_companies
"""

from __future__ import annotations

import re

# Trailing legal/entity tokens that don't distinguish companies.
_LEGAL_SUFFIXES = {"INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "PLC", "LLC", "SA", "NV"}


def normalize_company_name(name: str) -> tuple[str, ...]:
    """Reduce a company name to comparable tokens.

    'DELTA AIR LINES INC /DE/' and 'Delta Air Lines' both become
    ('DELTA', 'AIR', 'LINES').
    """
    text = name.upper().replace("’", "'").replace("'", "")
    text = re.sub(r"/[A-Z ]*/?\s*$", " ", text)  # state-of-incorporation tags like /DE/
    tokens = [t for t in re.split(r"[^0-9A-Z]+", text) if t]
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return tuple(tokens)


def match_document_companies(
    doc_names: list[str], companies: list[dict]
) -> tuple[dict[str, int], list[str]]:
    """Match Octus document company names to SimFin company ids by name tokens.

    Match tiers, most to least trustworthy — a unique match at a tier wins and
    later tiers are not consulted (so 'Dave Inc' can't make 'Dave & Buster's'
    ambiguous when a forward-prefix match exists):
    1. normalized tokens equal;
    2. document tokens are a prefix of company tokens ('JETBLUE' vs
       'JETBLUE AIRWAYS' — doc names are usually truncated legal names);
    3. company tokens are a prefix of document tokens.

    Returns:
        (name -> company id for unique matches, names left unmatched/ambiguous)
    """
    by_tokens = [(normalize_company_name(c["name"]), c["id"]) for c in companies]

    matched: dict[str, int] = {}
    unmatched: list[str] = []
    for doc_name in doc_names:
        doc_tokens = normalize_company_name(doc_name)
        if not doc_tokens:
            unmatched.append(doc_name)
            continue
        tiers = (
            [cid for tokens, cid in by_tokens if tokens == doc_tokens],
            [cid for tokens, cid in by_tokens if tokens[: len(doc_tokens)] == doc_tokens],
            [cid for tokens, cid in by_tokens if doc_tokens[: len(tokens)] == tokens],
        )
        for candidates in tiers:
            if len(set(candidates)) == 1:
                matched[doc_name] = candidates[0]
                break
        else:
            unmatched.append(doc_name)
    return matched, unmatched


def backfill_sectors(
    *,
    market: str | None = None,
    api_key: str | None = None,
    data_dir: str | None = None,
    database_url: str | None = None,
) -> int:
    """Fill companies.sector/industry from SimFin's industries dataset.

    Returns the number of companies updated.
    """
    import simfin as sf

    from core.config import settings
    from db.connection import get_connection
    from ingestion.simfin_ingest import _configure_simfin, _flatten, _opt_str

    _configure_simfin(api_key, data_dir)
    market = market or settings.simfin_market

    companies = _flatten(sf.load_companies(market=market))
    industries = _flatten(sf.load_industries())
    sector_by_industry_id = {
        _opt_str(row["IndustryId"]): (_opt_str(row["Sector"]), _opt_str(row["Industry"]))
        for row in industries.to_dict("records")
    }

    rows = []
    for row in companies.to_dict("records"):
        ticker = _opt_str(row.get("Ticker"))
        sector, industry = sector_by_industry_id.get(_opt_str(row.get("IndustryId")), (None, None))
        if ticker and (sector or industry):
            rows.append((sector, industry, ticker))

    with get_connection(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE companies ADD COLUMN IF NOT EXISTS industry TEXT")
            cur.executemany(
                "UPDATE companies SET sector = %s, industry = %s WHERE ticker = %s",
                rows,
            )
        conn.commit()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM companies WHERE sector IS NOT NULL")
            return int(cur.fetchone()["n"])


def backfill_document_companies(*, database_url: str | None = None) -> tuple[int, list[str]]:
    """Link documents to SimFin companies by normalized-name matching.

    Returns:
        (number of documents linked, document company names left unlinked)
    """
    from db.connection import get_connection

    with get_connection(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT company_name FROM documents WHERE company_name IS NOT NULL"
            )
            doc_names = [r["company_name"] for r in cur.fetchall()]
            cur.execute("SELECT id, name FROM companies")
            companies = list(cur.fetchall())

        matched, unmatched = match_document_companies(doc_names, companies)

        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE documents SET company_id = %s WHERE company_name = %s",
                [(cid, name) for name, cid in matched.items()],
            )
        conn.commit()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM documents WHERE company_id IS NOT NULL")
            return int(cur.fetchone()["n"]), unmatched


def main() -> None:
    with_sector = backfill_sectors()
    print(f"companies with sector: {with_sector}")
    linked, unmatched = backfill_document_companies()
    print(f"documents linked to a company: {linked}")
    if unmatched:
        print(f"unmatched document company names (left NULL): {unmatched}")


if __name__ == "__main__":
    main()
