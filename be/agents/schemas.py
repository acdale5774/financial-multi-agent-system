"""Pydantic models for the agent layer's traceable answers.

An answer is never just text: it carries every SQL query that was executed and
every tool call the agent made, in order, so any figure in the prose can be
traced back to the exact query (and rows) that produced it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SqlQueryRecord(BaseModel):
    """One executed read-only SQL query and what it returned."""

    query: str
    row_count: int
    rows: list[dict[str, Any]]  # JSON-safe copies of the returned rows


class ToolCallRecord(BaseModel):
    """One tool invocation in the agent loop (the audit trail)."""

    tool: str
    input: dict[str, Any]
    ok: bool
    summary: str  # short human-readable outcome, e.g. "8 rows" or the error


class AgentAnswer(BaseModel):
    """The agent's final, traceable response to a natural-language question."""

    question: str
    answer: str
    sql_queries: list[SqlQueryRecord]
    tool_calls: list[ToolCallRecord]
    model: str
    iterations: int
    stop_reason: str | None = None
