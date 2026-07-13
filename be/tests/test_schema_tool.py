"""Tests for the schema discovery tools (tools/schema_tool.py).

Database access is monkeypatched at the module boundary; these cover the
DB-free logic — table-name validation (which doubles as SQL-injection
protection), search-pattern construction, notes, and value truncation.
"""

from __future__ import annotations

import pytest

from tools import schema_tool


def test_describe_table_rejects_unknown_table(monkeypatch):
    monkeypatch.setattr(schema_tool, "_existing_tables", lambda db=None: {"companies"})
    calls = []
    monkeypatch.setattr(
        schema_tool, "run_read_only_sql", lambda *a, **k: calls.append(a) or []
    )

    with pytest.raises(ValueError, match="Unknown table"):
        schema_tool.describe_table('companies"; DROP TABLE companies; --')
    # Validation must happen before any SQL touches the (unverified) name.
    assert calls == []


def test_describe_table_merges_notes_and_truncates_samples(monkeypatch):
    monkeypatch.setattr(schema_tool, "_existing_tables", lambda db=None: {"financials"})

    def fake_sql(query, *, params=None, **kwargs):
        if "information_schema.columns" in query:
            return [
                {"column_name": "metric", "data_type": "text", "is_nullable": "NO"},
            ]
        return [{"metric": "x" * 500}]  # oversized sample value

    monkeypatch.setattr(schema_tool, "run_read_only_sql", fake_sql)
    result = schema_tool.describe_table("financials")

    assert result["columns"] == [{"name": "metric", "type": "text", "nullable": False}]
    assert "LONG/TALL" in result["notes"]  # curated quirks reach the agent
    sample_value = result["sample_rows"][0]["metric"]
    assert len(sample_value) <= schema_tool._SAMPLE_VALUE_MAX_CHARS + 1


def test_search_metrics_wraps_pattern_for_substring_match(monkeypatch):
    captured = {}

    def fake_sql(query, *, params=None, **kwargs):
        captured.update(params)
        return [{"metric": "revenue"}]

    monkeypatch.setattr(schema_tool, "run_read_only_sql", fake_sql)
    assert schema_tool.search_metrics("reven") == [{"metric": "revenue"}]
    assert captured["pat"] == "%reven%"


def test_resolve_company_passes_ticker_and_name_params(monkeypatch):
    captured = {}

    def fake_sql(query, *, params=None, **kwargs):
        captured.update(params)
        return []

    monkeypatch.setattr(schema_tool, "run_read_only_sql", fake_sql)
    schema_tool.resolve_company("Ford")
    assert captured == {"q": "Ford", "like": "%Ford%"}


def test_core_tables_have_curated_notes():
    # The semantic layer must at minimum cover what ingestion creates today.
    for table in ("companies", "financials", "documents", "document_chunks"):
        assert table in schema_tool._TABLE_NOTES
