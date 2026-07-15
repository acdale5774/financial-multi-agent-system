"""Schema discovery tools for the structured-data query agent.

The agent never gets the database schema hardcoded into its prompt. Instead it
discovers what exists at runtime through these tools:

- `list_tables` / `describe_table`: live introspection via information_schema,
  so a newly ingested table (the "more structured sources will arrive" case)
  is immediately discoverable with zero code changes.
- Curated `_TABLE_NOTES`: the semantic layer introspection can't express —
  join keys, unit conventions, fiscal-period quirks. Additive per source;
  a table with no note still works, it just lacks the extra guidance.
- `search_metrics`: in the long/tall `financials` table, metric names are DATA,
  not schema ('revenue', 'shares_diluted', ...). The agent must look them up
  rather than guess.
- `resolve_company`: entity resolution ("Ford" is stored as "Ford Motor";
  tickers are exact) plus data coverage, so the agent knows what years exist
  before writing a query against them.

All queries run through the read-only SQL tool, so discovery inherits the same
safety guarantees as agent-issued queries.
"""

from __future__ import annotations

from typing import Any

from tools.sql_tool import run_read_only_sql

# Truncate long cell values in sample rows (e.g. an embedding vector)
# so tool results stay small enough to feed back to the model.
_SAMPLE_VALUE_MAX_CHARS = 120

# Curated, source-specific semantics — the part of the catalog that lives with
# the code, not the database. Keyed by table name; unknown tables simply have
# no notes. When a new structured source is ingested, add one entry here.
_TABLE_NOTES: dict[str, str] = {
    "companies": (
        "One row per company; `ticker` is unique. `id` is the canonical join key "
        "(financials.company_id and documents.company_id reference it). Resolve "
        "names/tickers with the resolve_company tool instead of guessing spellings. "
        "`sector` (broad, e.g. 'Industrials') and `industry` (finer, e.g. 'Airlines') "
        "come from SimFin's industries taxonomy; a few companies have neither. "
        "There is NO CEO/officer data in the structured store."
    ),
    "financials": (
        "SimFin financials in LONG/TALL shape: one row per (company_id, statement, "
        "fiscal_year, fiscal_period, metric).\n"
        "- statement: 'income' | 'balance' | 'cashflow'.\n"
        "- fiscal_period: 'FY' (annual) or 'Q1'..'Q4'. Coverage quirks in this dataset:\n"
        "  * income and cashflow: FY and Q1-Q4 are loaded.\n"
        "  * balance: Q1-Q4 point-in-time snapshots only — there are no 'FY' balance\n"
        "    rows (SimFin convention: the Q4 snapshot IS the fiscal-year-end balance).\n"
        "- metric: snake_cased SimFin field names (e.g. 'revenue', 'net_income',\n"
        "  'shares_diluted'). Discover exact names with search_metrics — never guess.\n"
        "- There is NO eps metric: compute eps = net_income / shares_diluted in SQL\n"
        "  (NULLIF the denominator) and present it as computed.\n"
        "- value: raw currency units (391035000000 = $391.035B); `currency` is per row\n"
        "  and VARIES ACROSS COMPANIES — filter or group by currency in any\n"
        "  cross-company comparison or ranking.\n"
        "- Fiscal year can differ from calendar year; `report_date` is the period end."
    ),
    "documents": (
        "Metadata for the unstructured corpus (SEC filings + earnings-call "
        "transcripts). Useful via SQL for counts/coverage questions (e.g. how many "
        "10-Ks per company). `company_id` joins to companies (the cross-source key; "
        "NULL for the few documents whose company isn't in SimFin). The text itself "
        "lives in document_chunks."
    ),
    "document_chunks": (
        "Chunked + embedded document text for semantic search. The `embedding` "
        "column is an embedding vector — not meaningfully queryable via SQL; "
        "semantic retrieval goes through the document search tool, not this "
        "interface."
    ),
}


def _existing_tables(database_url: str | None = None) -> set[str]:
    """Names of all base tables in the public schema."""
    rows = run_read_only_sql(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """,
        database_url=database_url,
    )
    return {r["table_name"] for r in rows}


def list_tables(database_url: str | None = None) -> list[dict[str, Any]]:
    """List public tables with approximate row counts and a one-line summary."""
    rows = run_read_only_sql(
        """
        SELECT c.relname AS table_name,
               greatest(c.reltuples, 0)::bigint AS approx_rows
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'r'
        ORDER BY c.relname
        """,
        database_url=database_url,
    )
    return [
        {
            "table": r["table_name"],
            "approx_rows": int(r["approx_rows"]),
            "summary": _TABLE_NOTES.get(r["table_name"], "").split("\n")[0]
            or "(no curated notes)",
        }
        for r in rows
    ]


def describe_table(table: str, database_url: str | None = None) -> dict[str, Any]:
    """Columns, curated usage notes, and a few (truncated) sample rows.

    Raises:
        ValueError: if `table` is not an existing public table. Validating
            against the live catalog is also what makes the sample-row query
            safe to build with the (verified) table name inlined.
    """
    existing = _existing_tables(database_url)
    if table not in existing:
        raise ValueError(
            f"Unknown table '{table}'. Available: {', '.join(sorted(existing))}."
        )

    columns = run_read_only_sql(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %(t)s
        ORDER BY ordinal_position
        """,
        params={"t": table},
        database_url=database_url,
    )
    samples = run_read_only_sql(
        f'SELECT * FROM "{table}" LIMIT 3', database_url=database_url
    )
    return {
        "table": table,
        "columns": [
            {
                "name": c["column_name"],
                "type": c["data_type"],
                "nullable": c["is_nullable"] == "YES",
            }
            for c in columns
        ],
        "notes": _TABLE_NOTES.get(table, "(no curated notes for this table)"),
        "sample_rows": [
            {k: _short(v) for k, v in row.items()} for row in samples
        ],
    }


def search_metrics(pattern: str, database_url: str | None = None) -> list[dict[str, Any]]:
    """Search the `financials` metric catalog by case-insensitive substring.

    Returns matching metric names with their statement, data-point count,
    company coverage, and fiscal-year range — enough for the agent to pick the
    right metric and know whether the period it needs exists.
    """
    rows = run_read_only_sql(
        """
        SELECT metric, statement,
               count(*) AS data_points,
               count(DISTINCT company_id) AS companies,
               min(fiscal_year) AS first_year,
               max(fiscal_year) AS last_year
        FROM financials
        WHERE metric ILIKE %(pat)s
        GROUP BY metric, statement
        ORDER BY count(*) DESC
        LIMIT 40
        """,
        params={"pat": f"%{pattern}%"},
        statement_timeout_ms=15_000,
        database_url=database_url,
    )
    return list(rows)


def resolve_company(query: str, database_url: str | None = None) -> list[dict[str, Any]]:
    """Resolve a ticker or name fragment to companies with data coverage.

    Exact ticker matches rank first, then by how much financial data exists —
    SimFin carries thousands of companies, many with sparse coverage.
    """
    rows = run_read_only_sql(
        """
        SELECT c.id, c.ticker, c.name, c.sector,
               (upper(c.ticker) = upper(%(q)s)) AS exact_ticker,
               count(f.id) AS data_points,
               min(f.fiscal_year) AS first_year,
               max(f.fiscal_year) AS last_year
        FROM companies c
        LEFT JOIN financials f ON f.company_id = c.id
        WHERE upper(c.ticker) = upper(%(q)s) OR c.name ILIKE %(like)s
        GROUP BY c.id, c.ticker, c.name, c.sector
        ORDER BY (upper(c.ticker) = upper(%(q)s)) DESC, count(f.id) DESC
        LIMIT 10
        """,
        params={"q": query, "like": f"%{query}%"},
        statement_timeout_ms=15_000,
        database_url=database_url,
    )
    return list(rows)


def _short(value: Any) -> Any:
    """Truncate long values (embeddings, document text) for sample rows."""
    text = str(value)
    if len(text) > _SAMPLE_VALUE_MAX_CHARS:
        return text[:_SAMPLE_VALUE_MAX_CHARS] + "…"
    return value
