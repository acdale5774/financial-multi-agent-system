"""Tests for the agent tool-calling loop (agents/orchestrator.py).

The OpenAI client is faked so the loop's mechanics — dispatch, error
feedback, trace capture, iteration bounds, message shape — are tested without
network access or an API key. SQL execution is monkeypatched at the
orchestrator module boundary.
"""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

from agents import orchestrator
from agents.orchestrator import TOOL_DEFINITIONS, answer


def _tool_call(name: str, arguments: str, call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(
    text: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    finish_reason = finish_reason or ("tool_calls" if tool_calls else "stop")
    message = SimpleNamespace(content=text, tool_calls=tool_calls or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


class FakeClient:
    """OpenAI-shaped client that replays scripted responses."""

    def __init__(self, responses):
        self.requests: list[dict] = []
        responses_iter = iter(responses)

        def create(**kwargs):
            self.requests.append(kwargs)
            return next(responses_iter)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def test_direct_answer_without_tools():
    client = FakeClient([_response(text="Nothing to compute.")])
    result = answer("hello", client=client)

    assert result.answer == "Nothing to compute."
    assert result.sql_queries == []
    assert result.tool_calls == []
    assert result.iterations == 1
    assert result.stop_reason == "stop"
    # The loop must send the system prompt and the full tool surface.
    assert client.requests[0]["tools"] is TOOL_DEFINITIONS
    assert client.requests[0]["messages"][0]["role"] == "system"


def test_sql_round_trip_records_query_and_rows(monkeypatch):
    rows = [{"ticker": "F", "revenue": 184992000000.0}]
    monkeypatch.setattr(orchestrator, "run_read_only_sql", lambda q, **kw: rows)

    client = FakeClient(
        [
            _response(tool_calls=[_tool_call("run_sql", '{"query": "SELECT 1"}')]),
            _response(text="Ford: $185.0B (query 1)."),
        ]
    )
    result = answer("Ford revenue?", client=client)

    assert result.answer == "Ford: $185.0B (query 1)."
    assert len(result.sql_queries) == 1
    assert result.sql_queries[0].query == "SELECT 1"
    assert result.sql_queries[0].row_count == 1
    assert result.tool_calls[0].ok is True

    # Second request: system, user, assistant (with tool_calls), tool reply.
    second = client.requests[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assert second[2]["tool_calls"][0]["function"]["name"] == "run_sql"
    assert second[3]["tool_call_id"] == "call_1"
    assert json.loads(second[3]["content"])["row_count"] == 1
    # The history must be plain data — every tool call id gets answered and
    # the assistant turn round-trips through JSON (no SDK objects appended).
    json.dumps(second)


def test_sql_error_is_fed_back_not_raised(monkeypatch):
    def boom(query, **kwargs):
        raise ValueError("relation \"financialz\" does not exist")

    monkeypatch.setattr(orchestrator, "run_read_only_sql", boom)
    client = FakeClient(
        [
            _response(
                tool_calls=[_tool_call("run_sql", '{"query": "SELECT * FROM financialz"}')]
            ),
            _response(text="Recovered."),
        ]
    )
    result = answer("q", client=client)

    assert result.answer == "Recovered."
    assert result.tool_calls[0].ok is False
    assert "does not exist" in result.tool_calls[0].summary
    tool_msg = client.requests[1]["messages"][3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"].startswith("ERROR:")
    # Failed queries are not part of the citable audit trail.
    assert result.sql_queries == []


def test_malformed_tool_arguments_are_fed_back(monkeypatch):
    client = FakeClient(
        [
            _response(tool_calls=[_tool_call("run_sql", '{"query": "SELECT 1"')]),
            _response(text="Retried."),
        ]
    )
    result = answer("q", client=client)

    assert result.answer == "Retried."
    assert result.tool_calls[0].ok is False
    assert result.tool_calls[0].summary == "invalid tool arguments"
    tool_msg = client.requests[1]["messages"][3]
    assert tool_msg["content"].startswith("ERROR: Invalid JSON")
    assert result.sql_queries == []


def test_unknown_tool_returns_error_result():
    client = FakeClient(
        [
            _response(tool_calls=[_tool_call("send_email", '{"to": "x"}')]),
            _response(text="ok"),
        ]
    )
    result = answer("q", client=client)
    assert result.tool_calls[0].ok is False
    assert "Unknown tool" in result.tool_calls[0].summary


def test_parallel_tool_calls_each_get_a_tool_message(monkeypatch):
    monkeypatch.setattr(orchestrator, "list_tables", lambda db=None: [])
    monkeypatch.setattr(orchestrator, "search_metrics", lambda p, db=None: [])
    client = FakeClient(
        [
            _response(
                tool_calls=[
                    _tool_call("list_tables", "{}", "call_a"),
                    _tool_call("search_metrics", '{"pattern": "rev"}', "call_b"),
                ]
            ),
            _response(text="done"),
        ]
    )
    result = answer("q", client=client)

    second = client.requests[1]["messages"]
    tool_msgs = [m for m in second if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_a", "call_b"]
    assert len(result.tool_calls) == 2


def test_iteration_cap_stops_runaway_loop(monkeypatch):
    monkeypatch.setattr(orchestrator, "list_tables", lambda db=None: [])
    endless = (
        _response(tool_calls=[_tool_call("list_tables", "{}", f"call_{i}")])
        for i in itertools.count()
    )
    client = FakeClient(endless)
    result = answer("q", client=client, max_iterations=3)

    assert result.iterations == 3
    assert result.stop_reason == "tool_calls"
    assert "Stopped after 3 tool iterations" in result.answer


def test_content_filter_yields_explanatory_answer():
    client = FakeClient([_response(finish_reason="content_filter")])
    result = answer("q", client=client)
    assert result.stop_reason == "content_filter"
    assert "declined" in result.answer
