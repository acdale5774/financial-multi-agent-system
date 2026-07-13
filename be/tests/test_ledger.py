"""Tests for evidence minting (agents/ledger.py)."""

from __future__ import annotations

from decimal import Decimal

from agents.ledger import EvidenceLedger
from agents.schemas import SimFinCitation
from tools.document_search_tool import SearchResult


def _row(**extra):
    return {"ticker": "F", "fiscal_year": 2024, "fiscal_period": "FY", **extra}


def test_mints_one_citation_per_numeric_column():
    ledger = EvidenceLedger()
    rows, cite_maps, nudge = ledger.record_sql(
        "SELECT ...", [_row(revenue=Decimal("184992000000"), net_income=Decimal("5879000000"))]
    )

    assert nudge is None
    assert set(cite_maps[0]) == {"revenue", "net_income"}
    revenue = ledger.citations[cite_maps[0]["revenue"]]
    assert isinstance(revenue, SimFinCitation)
    assert revenue.ticker == "F"
    assert revenue.fiscal_year == 2024
    assert revenue.fiscal_period == "FY"
    assert revenue.metric == "revenue"
    assert revenue.value == 184992000000.0  # Decimal coerced before minting
    assert rows[0]["revenue"] == 184992000000.0
    assert ledger.sql_queries[0].row_count == 1


def test_long_shape_rows_use_metric_column_for_value():
    ledger = EvidenceLedger()
    _, cite_maps, _ = ledger.record_sql("q", [_row(metric="revenue", value=5.0)])
    citation = ledger.citations[cite_maps[0]["value"]]
    assert citation.metric == "revenue"


def test_identical_datapoint_reuses_id_across_queries():
    ledger = EvidenceLedger()
    _, first, _ = ledger.record_sql("q1", [_row(revenue=1.0)])
    _, second, _ = ledger.record_sql("q2", [_row(revenue=1.0)])
    assert first[0]["revenue"] == second[0]["revenue"]
    assert len(ledger.citations) == 1
    assert len(ledger.sql_queries) == 2  # both queries stay in the trace


def test_aggregate_rows_degrade_to_query_citation_with_nudge():
    ledger = EvidenceLedger()
    _, cite_maps, nudge = ledger.record_sql("SELECT count(*) ...", [{"count": 4599}])

    citation = ledger.citations[cite_maps[0]["*"]]
    assert citation.granularity == "query"
    assert "fiscal_period" in nudge


def test_document_minting_and_dedup():
    ledger = EvidenceLedger()
    result = SearchResult(
        document_id="octus-1",
        chunk_index=3,
        text="Risk factors " * 100,
        score=0.71234,
        title="Delta 10-K",
        metadata={"doc_type": "10-K", "company_name": "Delta Air Lines", "section": "Item 1A"},
    )
    first = ledger.record_documents("risks", {"doc_type": "10-K"}, [result])
    second = ledger.record_documents("risks again", None, [result])

    assert first[0].id == second[0].id == "D1"
    assert first[0].section == "Item 1A"
    assert len(first[0].snippet) <= 300
    assert ledger.searches[0]["filters"] == {"doc_type": "10-K"}


def test_cited_subset_dedupes_and_keeps_order():
    ledger = EvidenceLedger()
    _, cite_maps, _ = ledger.record_sql("q", [_row(revenue=1.0, eps=2.0)])
    ids = list(cite_maps[0].values())
    subset = ledger.cited_subset([ids[1], ids[0], ids[1], "S99"])
    assert [c.id for c in subset] == [ids[1], ids[0]]


def test_chart_registration_assigns_c_ids():
    ledger = EvidenceLedger()
    chart = ledger.record_chart(
        {"url": "/charts/x.png", "file": "/tmp/x.png", "spec": {"title": "t"}}, ["S1"]
    )
    assert chart.id == "C1"
    assert ledger.has("C1")


def test_citation_cap_appends_nudge_and_keeps_alignment():
    ledger = EvidenceLedger()
    rows = [
        _row(fiscal_period=f"Q{1 + i % 4}", fiscal_year=2000 + i, revenue=float(i))
        for i in range(70)
    ]
    safe_rows, cite_maps, nudge = ledger.record_sql("q", rows)

    assert len(cite_maps) == len(safe_rows) == 70
    assert cite_maps[0] and cite_maps[-1] == {}  # early rows cited, capped rows empty
    assert "Citation cap" in nudge


def test_values_for_query_granularity_exposes_row_cells():
    ledger = EvidenceLedger()
    _, cite_maps, _ = ledger.record_sql("q", [{"total": 61643000000.0, "n": 12}])
    assert set(ledger.values_for(cite_maps[0]["*"])) == {61643000000.0, 12.0}


def test_values_for_document_scales_unit_words():
    result = SearchResult(
        document_id="d", chunk_index=0, text="cash flow of $1.4 billion", score=0.5,
        title=None, metadata={},
    )
    ledger = EvidenceLedger()
    (cite,) = ledger.record_documents("q", None, [result])
    values = ledger.values_for(cite.id)
    assert 1.4 in values and 1.4e9 in values


def test_record_table_assigns_t_ids():
    from agents.schemas import TableCell

    ledger = EvidenceLedger()
    table = ledger.record_table("t", ["Quarter", "Revenue"], [[TableCell(text="Q1")]], "| |")
    assert table.id == "T1"
    assert ledger.has("T1")
