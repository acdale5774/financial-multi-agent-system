"""The structured-data query agent: a native OpenAI tool-calling loop.

`answer(question)` runs the loop:

    question -> model -> tool calls (discover / resolve / query) -> model
             -> ... -> final text + full audit trail (AgentAnswer)

Design decisions (see agents/README.md for the full rationale):
- **Native manual loop** rather than an agent framework: the loop is ~60
  lines, has no extra dependency, and every tool call passes through one
  place where it is recorded — which is the point, in an auditable
  financial context.
- **Tools over prompt-stuffing** for schema knowledge: the model discovers
  tables/metrics/companies at runtime, so new structured sources are usable
  the moment they're ingested.
- **Errors are feedback, not failures**: a bad SQL statement comes back to
  the model as an "ERROR: ..." tool result and it self-corrects, bounded by
  `max_iterations`.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from agents.prompts import SYSTEM_PROMPT
from agents.schemas import AgentAnswer, SqlQueryRecord, ToolCallRecord
from tools.schema_tool import describe_table, list_tables, resolve_company, search_metrics
from tools.sql_tool import run_read_only_sql

# Hard cap on rows fetched per agent query; the model is told when it's hit.
MAX_SQL_ROWS = 200
# Rows actually shown back to the model (full set is kept in the trace).
ROWS_SHOWN_TO_MODEL = 50
# Server-side timeout for agent-issued SQL.
SQL_TIMEOUT_MS = 15_000

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": (
                "List every table in the financial database with approximate row "
                "counts and a one-line summary. Start here when you are unsure "
                "what data exists."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": (
                "Get a table's columns and types, curated usage notes (join keys, "
                "unit conventions, period-labeling quirks), and a few sample rows. "
                "Always describe a table before querying it for the first time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "A table name from list_tables."}
                },
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_metrics",
            "description": (
                "Search the metric catalog of the long/tall `financials` table by "
                "case-insensitive substring. Returns exact metric names with their "
                "statement, data-point count, company coverage, and fiscal-year "
                "range. Metric names are data, not schema — always look them up "
                "here instead of guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Substring to match, e.g. 'revenue', 'shares', 'cash'.",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_company",
            "description": (
                "Resolve a company name or ticker to its database id, canonical "
                "name, and data coverage (fiscal-year range, data-point count). "
                "Always resolve before filtering by company — stored names rarely "
                "match the question verbatim. Returns up to 10 ranked candidates; "
                "pick by NAME, not rank alone (the ticker 'FORD' is Forward "
                "Industries — Ford Motor is ticker 'F')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Ticker or company-name fragment."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Execute ONE read-only SELECT (or WITH … SELECT) against Postgres. "
                "Writes, DDL, and multiple statements are rejected. Aggregate and "
                "LIMIT in SQL — results are capped at "
                f"{MAX_SQL_ROWS} rows. Every query is recorded verbatim and shown "
                "to the user as the answer's audit trail, so write clear SQL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A single SELECT statement."}
                },
                "required": ["query"],
            },
        },
    },
]


def answer(
    question: str,
    *,
    model: str | None = None,
    max_iterations: int | None = None,
    database_url: str | None = None,
    client: Any = None,
) -> AgentAnswer:
    """Answer a natural-language question over the structured store.

    Args:
        question: The user's question.
        model: OpenAI model id; defaults to settings.agent_model.
        max_iterations: Cap on model turns (each may carry several tool calls).
        database_url: Override Postgres connection string.
        client: Injectable OpenAI-compatible client (tests use a fake).

    Returns:
        AgentAnswer with the prose answer plus the full SQL/tool audit trail.
    """
    from core.config import settings

    model = model or settings.agent_model
    max_iterations = max_iterations or settings.agent_max_iterations
    if client is None:
        import openai  # lazy: unit tests inject a fake client instead

        client = openai.OpenAI(api_key=settings.openai_api_key or None)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    sql_queries: list[SqlQueryRecord] = []
    tool_calls: list[ToolCallRecord] = []
    finish_reason: str | None = None
    message = None

    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=16000,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )
        choice = response.choices[0]
        finish_reason = choice.finish_reason
        message = choice.message
        requested = message.tool_calls or []
        if finish_reason != "tool_calls" or not requested:
            break

        # Append the assistant turn (as a plain dict — it must round-trip
        # through JSON), then answer EVERY tool_call_id with its own
        # role="tool" message, in order.
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in requested
                ],
            }
        )
        for tc in requested:
            try:
                tool_input = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                tool_input = {"raw_arguments": tc.function.arguments}
                content, ok, summary = (
                    f"Invalid JSON in tool arguments: {exc}",
                    False,
                    "invalid tool arguments",
                )
            else:
                content, ok, summary = _execute_tool(
                    tc.function.name, tool_input, sql_queries, database_url
                )
            tool_calls.append(
                ToolCallRecord(tool=tc.function.name, input=tool_input, ok=ok, summary=summary)
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content if ok else f"ERROR: {content}",
                }
            )

    answer_text = ((message.content if message is not None else "") or "").strip()
    if finish_reason == "tool_calls":
        answer_text = (
            answer_text
            or f"Stopped after {iterations} tool iterations without a final answer; "
            "the executed queries below show the progress made."
        )
    elif finish_reason == "content_filter":
        answer_text = answer_text or "The model declined to answer this question."

    return AgentAnswer(
        question=question,
        answer=answer_text,
        sql_queries=sql_queries,
        tool_calls=tool_calls,
        model=model,
        iterations=iterations,
        stop_reason=finish_reason,
    )


def _execute_tool(
    name: str,
    tool_input: dict[str, Any],
    sql_queries: list[SqlQueryRecord],
    database_url: str | None,
) -> tuple[str, bool, str]:
    """Run one tool call; never raises.

    Returns:
        (content, ok, summary) — `content` goes back to the model as the
        tool message (JSON on success, the error message on failure so the
        model can self-correct), `summary` goes into the audit trail.
    """
    try:
        if name == "run_sql":
            query = tool_input["query"]
            rows = [
                _jsonable_row(r)
                for r in run_read_only_sql(
                    query,
                    max_rows=MAX_SQL_ROWS,
                    statement_timeout_ms=SQL_TIMEOUT_MS,
                    database_url=database_url,
                )
            ]
            sql_queries.append(
                SqlQueryRecord(query=query, row_count=len(rows), rows=rows)
            )
            payload: dict[str, Any] = {"row_count": len(rows), "rows": rows[:ROWS_SHOWN_TO_MODEL]}
            if len(rows) > ROWS_SHOWN_TO_MODEL:
                payload["note"] = (
                    f"Showing first {ROWS_SHOWN_TO_MODEL} of {len(rows)} rows — "
                    "aggregate or LIMIT in SQL to narrow the result."
                )
            if len(rows) == MAX_SQL_ROWS:
                payload["note"] = (
                    f"Row cap of {MAX_SQL_ROWS} hit; the true result may be larger. "
                    "Aggregate in SQL instead of enumerating rows."
                )
            return json.dumps(payload, default=str), True, f"{len(rows)} rows"
        if name == "list_tables":
            result = list_tables(database_url)
            return json.dumps(result, default=str), True, f"{len(result)} tables"
        if name == "describe_table":
            result = describe_table(tool_input["table"], database_url)
            return (
                json.dumps(result, default=str),
                True,
                f"{tool_input['table']}: {len(result['columns'])} columns",
            )
        if name == "search_metrics":
            result = search_metrics(tool_input["pattern"], database_url)
            return json.dumps(result, default=str), True, f"{len(result)} metrics"
        if name == "resolve_company":
            result = resolve_company(tool_input["query"], database_url)
            return json.dumps(result, default=str), True, f"{len(result)} matches"
        raise ValueError(f"Unknown tool: {name}")
    except Exception as exc:  # noqa: BLE001 - errors are fed back to the model
        message = f"{type(exc).__name__}: {exc}"
        return message, False, message[:200]


def _jsonable_row(row: dict[str, Any]) -> dict[str, Any]:
    """Coerce psycopg row values (Decimal, date, ...) to JSON-safe types."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Decimal):
            out[key] = float(value)
        elif isinstance(value, (date, datetime)):
            out[key] = value.isoformat()
        else:
            out[key] = value
    return out
