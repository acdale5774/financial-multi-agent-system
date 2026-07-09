"""Safe, read-only SQL access to structured financial data.

This tool lets the agent query SimFin financials in Postgres. Because the query
text may be influenced by an LLM, **read-only enforcement is a hard requirement**,
not a nicety.
"""

from __future__ import annotations

from typing import Any

# One result row as a column-name -> value mapping.
SqlRow = dict[str, Any]


def run_read_only_sql(
    query: str,
    *,
    params: dict[str, Any] | None = None,
    max_rows: int = 1000,
    database_url: str | None = None,
) -> list[SqlRow]:
    """Execute a read-only SQL query and return the rows.

    Args:
        query: A single SELECT statement.
        params: Parameter values for the query (use bound params, never string
            interpolation).
        max_rows: Hard cap on returned rows to bound response size.
        database_url: Postgres connection string; falls back to DATABASE_URL.

    Returns:
        A list of rows as dicts (empty list if no rows).

    Raises:
        ValueError: If the statement is not a single, read-only SELECT.

    Safety model (to implement):
        - Connect using a Postgres role granted SELECT only (defense in depth).
        - Run inside a `READ ONLY` transaction.
        - Reject anything that isn't exactly one SELECT (no DML/DDL, no multiple
          statements, no CTE-wrapped writes).
        - Apply a statement timeout and enforce `max_rows` via LIMIT.

    TODO: Implement validation + execution with psycopg. Prefer an allow-list /
          parsed-AST check over regex for statement-type enforcement.
    """
    raise NotImplementedError
