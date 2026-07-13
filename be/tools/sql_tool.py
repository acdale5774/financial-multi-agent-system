"""Safe, read-only SQL access to structured financial data.

This tool lets the agent query SimFin financials in Postgres. Because the query
text may be influenced by an LLM, read-only enforcement is a hard requirement.

Defense in depth (two independent layers):
  1. Statement validation (`_assert_read_only`) — only a single SELECT / WITH…SELECT
     is allowed; write keywords and multiple statements are rejected.
  2. A read-only transaction on the connection (`conn.read_only = True`) plus a
     `statement_timeout`, so even a validation miss cannot mutate data.

For production, also connect using a Postgres role granted SELECT only — that
makes read-only enforcement true at the database boundary, not just in code.
"""

from __future__ import annotations

import re
from typing import Any

# One result row as a column-name -> value mapping.
SqlRow = dict[str, Any]

# Statement leading keywords we allow.
_ALLOWED_STARTS = frozenset({"select", "with"})

# Keywords that indicate a write / DDL / side effect. Matched as whole words.
_FORBIDDEN_KEYWORDS = frozenset(
    {
        "insert", "update", "delete", "merge", "upsert",
        "drop", "alter", "create", "truncate", "rename",
        "grant", "revoke", "comment", "copy", "call", "do",
        "vacuum", "analyze", "refresh", "reindex", "cluster",
        "lock", "set", "reset", "begin", "commit", "rollback",
    }
)


class ReadOnlyViolation(ValueError):
    """Raised when a query is not a single, read-only SELECT."""


def _assert_read_only(query: str) -> str:
    """Validate that `query` is a single read-only SELECT and return it cleaned.

    Raises:
        ReadOnlyViolation: if the query is empty, multi-statement, not a SELECT,
            or contains a forbidden (write/DDL) keyword.
    """
    stripped = query.strip().rstrip(";").strip()
    if not stripped:
        raise ReadOnlyViolation("Empty query.")
    if ";" in stripped:
        raise ReadOnlyViolation("Multiple statements are not allowed.")

    lowered = stripped.lower()
    leading = re.match(r"[a-z]+", lowered)
    if leading is None or leading.group(0) not in _ALLOWED_STARTS:
        raise ReadOnlyViolation("Only SELECT (or WITH … SELECT) queries are allowed.")

    # Whole-word tokens (underscores kept together, so `created_at` != `create`).
    tokens = set(re.findall(r"[a-z_]+", lowered))
    forbidden = tokens & _FORBIDDEN_KEYWORDS
    if forbidden:
        raise ReadOnlyViolation(
            "Disallowed keyword(s): " + ", ".join(sorted(forbidden)) + "."
        )
    return stripped


def run_read_only_sql(
    query: str,
    *,
    params: dict[str, Any] | None = None,
    max_rows: int = 1000,
    statement_timeout_ms: int = 5000,
    database_url: str | None = None,
) -> list[SqlRow]:
    """Execute a validated, read-only SELECT and return up to `max_rows` rows.

    Args:
        query: A single SELECT statement (may use named `%(name)s` params).
        params: Parameter values, bound safely (never string-interpolated).
        max_rows: Hard cap on returned rows.
        statement_timeout_ms: Server-side timeout to bound runaway queries.
        database_url: Override Postgres connection string.

    Returns:
        Rows as dicts (empty list if none).

    Raises:
        ReadOnlyViolation: if the statement is not a single read-only SELECT.
    """
    safe_query = _assert_read_only(query)

    # Imported lazily so the validation logic above stays unit-testable without
    # a database driver installed.
    from db.connection import get_connection

    with get_connection(database_url) as conn:
        conn.read_only = True  # transaction-level guard (set before any statement)
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")
            cur.execute(safe_query, params)
            rows = cur.fetchmany(max_rows)
    return list(rows)
