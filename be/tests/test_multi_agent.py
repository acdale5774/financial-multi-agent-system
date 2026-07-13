"""Tests for the routed multi-agent pipeline (agents/multi_agent.py).

LangChain's GenericFakeChatModel drives real create_agent graphs offline (a
bind_tools no-op subclass makes it tool-capable); DB and search functions are
monkeypatched at the lc_tools module boundaries. This exercises the actual
LangChain execution path — tool dispatch, message flow, repair turns,
recursion caps — without a network or API key.
"""

from __future__ import annotations

import itertools
import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agents import lc_tools, multi_agent
from agents.multi_agent import Route, answer_multi, dispatch
from agents.schemas import DocumentCitation, SimFinCitation
from tools import chart_tool, document_search_tool, schema_tool
from tools.document_search_tool import SearchResult


class FakeToolModel(GenericFakeChatModel):
    """Fake chat model that create_agent can bind tools to."""

    def bind_tools(self, tools, **kwargs):
        return self


def _model(*messages):
    return FakeToolModel(messages=iter(messages))


def _tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _route(name: str):
    return lambda question: Route(route=name, reason="scripted")


FORD_ROW = {
    "ticker": "F",
    "fiscal_year": 2024,
    "fiscal_period": "FY",
    "revenue": 184992000000.0,
}


def test_quant_flow_cites_minted_ids(monkeypatch):
    monkeypatch.setattr(lc_tools, "run_read_only_sql", lambda q, **kw: [dict(FORD_ROW)])
    model = _model(
        _tool_call("query_financials", {"query": "SELECT ..."}),
        AIMessage(content="Ford revenue was $185.0B in FY2024 [S1]."),
    )

    result = answer_multi("Ford revenue?", chat_model=model, router=_route("quant"))

    assert result.route == "quant"
    assert "[S1]" in result.answer
    assert [c.id for c in result.citations] == ["S1"]
    citation = result.citations[0]
    assert isinstance(citation, SimFinCitation)
    assert (citation.ticker, citation.fiscal_period, citation.metric) == ("F", "FY", "revenue")
    assert result.sql_queries[0].query == "SELECT ..."
    assert result.validation.markers_found == 1
    assert result.validation.invalid_markers_stripped == []
    assert result.validation.value_mismatch_sentences == 0


def test_docs_flow_cites_document_chunks(monkeypatch):
    chunk = SearchResult(
        document_id="octus-9",
        chunk_index=2,
        text="Fuel cost volatility remains a principal risk.",
        score=0.66,
        title="Delta Air Lines 10-K",
        metadata={"doc_type": "10-K", "company_name": "Delta Air Lines", "section": "Item 1A"},
    )
    monkeypatch.setattr(document_search_tool, "search_documents", lambda *a, **k: [chunk])
    model = _model(
        _tool_call("search_documents", {"query": "risks", "doc_type": "10-K"}),
        AIMessage(content="Delta flags fuel cost volatility as a principal risk [D1]."),
    )

    result = answer_multi("Delta risks?", chat_model=model, router=_route("docs"))

    assert result.route == "docs"
    assert isinstance(result.citations[0], DocumentCitation)
    assert result.citations[0].section == "Item 1A"
    assert result.searches[0]["result_count"] == 1


def test_chart_values_are_hydrated_from_ledger(monkeypatch, tmp_path):
    rows = [
        {"ticker": "F", "fiscal_year": 2024, "fiscal_period": "Q1", "eps": 0.33},
        {"ticker": "F", "fiscal_year": 2024, "fiscal_period": "Q2", "eps": 0.46},
    ]
    monkeypatch.setattr(lc_tools, "run_read_only_sql", lambda q, **kw: [dict(r) for r in rows])
    monkeypatch.setattr(chart_tool, "CHART_DIR", tmp_path)
    model = _model(
        _tool_call("query_financials", {"query": "SELECT eps..."}),
        _tool_call(
            "create_chart",
            {
                "title": "Ford EPS",
                "kind": "line",
                "x": ["Q1 2024", "Q2 2024"],
                "series": [{"name": "EPS", "citation_ids": ["S1", "S2"]}],
            },
            "c2",
        ),
        AIMessage(content="EPS rose from 0.33 [S1] to 0.46 [S2]. ![Ford EPS](/charts/x)"),
    )

    result = answer_multi("Plot Ford EPS", chat_model=model, router=_route("quant"))

    assert len(result.charts) == 1
    chart = result.charts[0]
    assert chart.id == "C1"
    assert chart.spec["series"][0]["values"] == [0.33, 0.46]  # ledger values, not model text
    assert chart.citation_ids == ["S1", "S2"]


def test_chart_with_unknown_citation_id_errors_and_agent_recovers(monkeypatch, tmp_path):
    monkeypatch.setattr(
        lc_tools, "run_read_only_sql", lambda q, **kw: [dict(FORD_ROW)]
    )
    monkeypatch.setattr(chart_tool, "CHART_DIR", tmp_path)
    model = _model(
        _tool_call("query_financials", {"query": "SELECT ..."}),
        _tool_call(
            "create_chart",
            {
                "title": "t",
                "kind": "line",
                "x": ["FY 2024"],
                "series": [{"name": "rev", "citation_ids": ["S99"]}],
            },
            "c2",
        ),
        AIMessage(content="Could not chart; revenue was $185.0B in FY2024 [S1]."),
    )

    result = answer_multi("chart?", chat_model=model, router=_route("quant"))

    assert result.charts == []  # bad id produced an ERROR tool message, no chart
    assert result.citations[0].id == "S1"


def test_invalid_marker_triggers_one_repair_turn(monkeypatch):
    monkeypatch.setattr(lc_tools, "run_read_only_sql", lambda q, **kw: [dict(FORD_ROW)])
    model = _model(
        _tool_call("query_financials", {"query": "SELECT ..."}),
        AIMessage(content="Revenue was $185.0B in FY2024 [S7]."),  # invalid id
        AIMessage(content="Revenue was $185.0B in FY2024 [S1]."),  # repair turn
    )

    result = answer_multi("Ford revenue?", chat_model=model, router=_route("quant"))

    assert result.validation.repaired is True
    assert "[S1]" in result.answer
    assert result.validation.invalid_markers_stripped == []


def test_marker_still_invalid_after_repair_is_stripped(monkeypatch):
    monkeypatch.setattr(lc_tools, "run_read_only_sql", lambda q, **kw: [dict(FORD_ROW)])
    model = _model(
        _tool_call("query_financials", {"query": "SELECT ..."}),
        AIMessage(content="Revenue was $185.0B [S7]."),
        AIMessage(content="Revenue was $185.0B [S7]."),  # repair didn't help
    )

    result = answer_multi("q", chat_model=model, router=_route("quant"))

    assert "[S7]" not in result.answer
    assert result.validation.invalid_markers_stripped == ["S7"]
    assert result.validation.repaired is True


def test_recursion_cap_degrades_to_partial_answer(monkeypatch):
    monkeypatch.setattr(schema_tool, "list_tables", lambda db=None: [])
    endless = (
        _tool_call("list_tables", {}, f"c{i}") for i in itertools.count()
    )
    model = FakeToolModel(messages=endless)

    result = answer_multi(
        "q", chat_model=model, router=_route("quant"), max_iterations=2
    )

    assert "iteration cap" in result.answer
    assert result.citations == []


def test_unsupported_route_answers_without_specialist():
    result = answer_multi(
        "Write me a poem", chat_model=object(), router=_route("unsupported")
    )
    assert result.route == "unsupported"
    assert "financial" in result.answer
    assert result.sql_queries == []


def test_dispatch_falls_back_to_hybrid_on_error():
    class Broken:
        def with_structured_output(self, schema):
            raise RuntimeError("no api key")

    route = dispatch("anything", chat_model=Broken())
    assert route.route == "hybrid"
    assert "defaulted" in route.reason


def test_query_financials_tool_injects_cite_ids(monkeypatch):
    monkeypatch.setattr(lc_tools, "run_read_only_sql", lambda q, **kw: [dict(FORD_ROW)])
    ledger = multi_agent.EvidenceLedger()
    tools = {t.name: t for t in lc_tools.make_toolsets(ledger)["quant"]}

    payload = json.loads(tools["query_financials"].invoke({"query": "SELECT 1"}))

    assert payload["rows"][0]["cites"] == {"revenue": "S1"}
    assert payload["row_count"] == 1
