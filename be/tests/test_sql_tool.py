"""Unit tests for read-only SQL enforcement (no database required).

These exercise the pure validation layer `_assert_read_only`; the execution path
against a live Postgres is covered by the integration smoke test.
"""

from __future__ import annotations

import pytest

from tools.sql_tool import ReadOnlyViolation, _assert_read_only


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM financials",
        "  select value from financials where metric = 'revenue' ;",
        "WITH x AS (SELECT 1 AS n) SELECT n FROM x",
        "SELECT created_at FROM companies",  # 'created_at' must not trip 'create'
    ],
)
def test_allows_read_only_selects(query: str) -> None:
    assert _assert_read_only(query)


@pytest.mark.parametrize(
    "query",
    [
        "",
        "INSERT INTO companies (ticker) VALUES ('X')",
        "UPDATE financials SET value = 0",
        "DELETE FROM financials",
        "DROP TABLE companies",
        "TRUNCATE financials",
        "SELECT 1; DROP TABLE companies",  # multiple statements
        "WITH x AS (INSERT INTO companies VALUES ('a') RETURNING id) SELECT * FROM x",
        "GRANT ALL ON financials TO public",
    ],
)
def test_rejects_writes_and_multi_statements(query: str) -> None:
    with pytest.raises(ReadOnlyViolation):
        _assert_read_only(query)
