"""Pydantic models for the agent layer's traceable answers.

An answer is never just text: it carries every SQL query that was executed,
every tool call the agent made, and — in the multi-agent system — a typed
citation record for every [S#]/[D#] marker in the prose, so any claim can be
traced back to the exact SimFin data point or document chunk behind it.
"""

from __future__ import annotations

from typing import Any, Literal

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


# ---------------------------------------------------------------------------
# Multi-agent (requirement 3): citations, charts, and the cited answer
# ---------------------------------------------------------------------------


class SimFinCitation(BaseModel):
    """One citable SimFin data point (or, degraded, one whole query).

    Minted by the evidence ledger from actual SQL result rows — never written
    by the model. Computed figures (EPS, growth) are minted from the rows of
    the query that computed them, so the formula is visible in the verbatim
    SQL at `sql_index`.
    """

    id: str
    source: Literal["simfin"] = "simfin"
    granularity: Literal["datapoint", "query"] = "datapoint"
    ticker: str | None = None
    company: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    metric: str | None = None
    value: float | None = None
    currency: str | None = None
    # Index into the answer's sql_queries — the verbatim SQL this came from.
    sql_index: int


class DocumentCitation(BaseModel):
    """One citable document chunk (SEC filing or transcript)."""

    id: str
    source: Literal["document"] = "document"
    document_id: str
    title: str | None = None
    doc_type: str | None = None
    company_name: str | None = None
    section: str | None = None
    chunk_index: int
    document_date: str | None = None
    snippet: str  # deterministic excerpt of the chunk, captured by code
    score: float


Citation = SimFinCitation | DocumentCitation


class ChartRecord(BaseModel):
    """A rendered chart plus the citations its numbers were hydrated from."""

    id: str
    source: Literal["chart"] = "chart"
    url: str
    file: str
    spec: dict[str, Any]
    citation_ids: list[str] = []


class TableCell(BaseModel):
    """One table cell: either a label or a citation-hydrated value."""

    text: str | None = None
    value: float | None = None
    citation_id: str | None = None


class TableRecord(BaseModel):
    """A structured multi-metric table; every value cell traces to evidence."""

    id: str
    source: Literal["table"] = "table"
    title: str
    columns: list[str]
    rows: list[list[TableCell]]
    markdown: str  # the same table rendered for the prose answer


class ValidationReport(BaseModel):
    """What the deterministic citation validator found (and fixed)."""

    markers_found: int = 0
    invalid_markers_stripped: list[str] = []
    repaired: bool = False
    # Warning-severity lints (see agents/citations.py for why not failures).
    uncited_numeric_sentences: int = 0
    value_mismatch_sentences: int = 0
    warnings: list[str] = []


class MultiAgentAnswer(BaseModel):
    """The multi-agent system's cited, traceable response."""

    question: str
    route: str  # 'quant' | 'docs' | 'hybrid' | 'unsupported'
    route_reason: str
    answer: str  # markdown with inline [S#]/[D#] markers
    citations: list[Citation]  # only records actually cited, in prose order
    charts: list[ChartRecord]
    tables: list[TableRecord] = []
    validation: ValidationReport
    sql_queries: list[SqlQueryRecord]
    searches: list[dict[str, Any]]
    model: str
