"""The evidence ledger: citation records minted by code, never by the model.

Every evidence-bearing tool call (SQL, document search, chart render) passes
through a per-request `EvidenceLedger`, which mints citation IDs (S# for
SimFin data points, D# for document chunks, C# for charts) and stores the full
records. The model only ever sees and repeats the IDs — it cannot fabricate a
citation record — and a deterministic validator (agents/citations.py) checks
every marker in the final prose against the ledger before the answer leaves
the API.

SimFin rows become citable data points when the SQL projects the citation-key
columns (ticker/company, fiscal_year, fiscal_period); every other numeric
column in such a row is minted as one data point, so a pivot row with
revenue + net_income + eps yields three citations. Rows without those keys
(e.g. aggregates like a company count) fall back to one query-granularity
citation, keeping the claim traceable to the verbatim SQL.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from agents.schemas import (
    ChartRecord,
    Citation,
    DocumentCitation,
    SimFinCitation,
    SqlQueryRecord,
    TableCell,
    TableRecord,
)
from tools.document_search_tool import SearchResult

# How many data-point citations a single query may mint (keeps tool results
# and answer payloads bounded; the full rows stay in sql_queries regardless).
MAX_CITATIONS_PER_QUERY = 60

# Column-name aliases for the citation keys, checked in order.
_TICKER_COLS = ("ticker", "symbol")
_COMPANY_COLS = ("company", "company_name", "name")
_YEAR_COLS = ("fiscal_year", "year", "fy")
_PERIOD_COLS = ("fiscal_period", "period", "quarter")
_CURRENCY_COLS = ("currency",)
# Numeric columns never worth citing as data points.
_NON_METRIC_COLS = {"id", "company_id", "document_id", "chunk_index", "sql_index"} | set(
    _YEAR_COLS
)

_SNIPPET_CHARS = 300


class EvidenceLedger:
    """Per-request evidence store; mints S#/D#/C# ids at tool boundaries."""

    def __init__(self) -> None:
        self.citations: dict[str, Citation] = {}
        self.charts: dict[str, ChartRecord] = {}
        self.tables: dict[str, TableRecord] = {}
        self.sql_queries: list[SqlQueryRecord] = []
        self.searches: list[dict[str, Any]] = []
        # Dedup: identical evidence re-retrieved later reuses its id.
        self._by_key: dict[tuple, str] = {}
        # Full chunk text per D-id (validation-only; responses carry snippets).
        self._doc_texts: dict[str, str] = {}
        self._doc_values: dict[str, list[float]] = {}

    # -- SimFin ---------------------------------------------------------------

    def record_sql(
        self, query: str, rows: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]], str | None]:
        """Record an executed query and mint citations from its rows.

        Returns:
            (json-safe rows, per-row {column -> citation id} maps aligned with
            the rows, optional nudge for the model when no data-point
            citations could be minted).
        """
        rows = [jsonable_row(r) for r in rows]
        self.sql_queries.append(SqlQueryRecord(query=query, row_count=len(rows), rows=rows))
        sql_index = len(self.sql_queries) - 1

        cite_maps: list[dict[str, str]] = []
        minted_total = 0
        capped = False
        for row in rows:
            if minted_total >= MAX_CITATIONS_PER_QUERY:
                capped = True
                cite_maps.append({})
                continue
            per_column = self._mint_row(row, sql_index)
            minted_total += len(per_column)
            cite_maps.append({col: c.id for col, c in per_column.items()})

        if minted_total:
            nudge = (
                f"Citation cap ({MAX_CITATIONS_PER_QUERY} data points per query) "
                "reached — later rows have no cites. Aggregate or LIMIT in SQL "
                "if you need to cite figures from them."
                if capped
                else None
            )
            return rows, cite_maps, nudge

        # Aggregate / non-keyed result: keep it traceable at query granularity.
        fallback = self._add(
            SimFinCitation(id="", granularity="query", sql_index=sql_index),
            key=("query", sql_index),
        )
        cite_maps = [{"*": fallback.id} for _ in rows] or [{"*": fallback.id}]
        nudge = (
            "Rows lack the citation keys (ticker or company, plus fiscal_year "
            f"and fiscal_period), so one query-level citation [{fallback.id}] "
            "covers this result. For per-figure citations, project those "
            "columns in the SELECT."
            if rows
            else None
        )
        return rows, cite_maps, nudge

    def _mint_row(self, row: dict[str, Any], sql_index: int) -> dict[str, SimFinCitation]:
        ticker = _pick_str(row, _TICKER_COLS)
        company = _pick_str(row, _COMPANY_COLS)
        year = _pick_int(row, _YEAR_COLS)
        period = _pick_str(row, _PERIOD_COLS)
        if not (ticker or company) or year is None or period is None:
            return {}

        period = period.upper()
        currency = _pick_str(row, _CURRENCY_COLS)
        explicit_metric = _pick_str(row, ("metric",))

        minted: dict[str, SimFinCitation] = {}
        for col, raw in row.items():
            if col.lower() in _NON_METRIC_COLS or isinstance(raw, bool):
                continue
            if not isinstance(raw, (int, float)):
                continue
            metric = explicit_metric if col.lower() == "value" and explicit_metric else col
            value = float(raw)
            key = ("simfin", ticker or company, year, period, metric, value)
            minted[col] = self._add(
                SimFinCitation(
                    id="",
                    ticker=ticker,
                    company=company,
                    fiscal_year=year,
                    fiscal_period=period,
                    metric=metric,
                    value=value,
                    currency=currency,
                    sql_index=sql_index,
                ),
                key=key,
            )
        return minted

    # -- Documents ------------------------------------------------------------

    def record_documents(
        self, query: str, filters: dict[str, Any] | None, results: list[SearchResult]
    ) -> list[DocumentCitation]:
        """Record a document search and mint one citation per chunk."""
        self.searches.append(
            {"query": query, "filters": filters or {}, "result_count": len(results)}
        )
        minted = []
        for r in results:
            meta = r.metadata or {}
            citation = DocumentCitation(
                id="",
                document_id=r.document_id,
                title=r.title,
                doc_type=meta.get("doc_type"),
                company_name=meta.get("company_name"),
                section=meta.get("section"),
                chunk_index=r.chunk_index,
                document_date=meta.get("document_date"),
                snippet=r.text[:_SNIPPET_CHARS],
                score=round(r.score, 4),
            )
            added = self._add(citation, key=("doc", r.document_id, r.chunk_index))
            self._doc_texts.setdefault(added.id, r.text)
            minted.append(added)
        return minted

    # -- Charts ---------------------------------------------------------------

    def record_chart(self, rendered: dict[str, Any], citation_ids: list[str]) -> ChartRecord:
        """Register a successfully rendered chart (see tools/chart_tool.py)."""
        chart = ChartRecord(
            id=f"C{len(self.charts) + 1}",
            url=rendered["url"],
            file=rendered["file"],
            spec=rendered["spec"],
            citation_ids=citation_ids,
        )
        self.charts[chart.id] = chart
        return chart

    # -- Tables ---------------------------------------------------------------

    def record_table(
        self, title: str, columns: list[str], rows: list[list[TableCell]], markdown: str
    ) -> TableRecord:
        """Register a citation-hydrated table (see lc_tools.create_table)."""
        table = TableRecord(
            id=f"T{len(self.tables) + 1}",
            title=title,
            columns=columns,
            rows=rows,
            markdown=markdown,
        )
        self.tables[table.id] = table
        return table

    # -- Lookup ---------------------------------------------------------------

    def has(self, citation_id: str) -> bool:
        return (
            citation_id in self.citations
            or citation_id in self.charts
            or citation_id in self.tables
        )

    def get_simfin(self, citation_id: str) -> SimFinCitation | None:
        record = self.citations.get(citation_id)
        return record if isinstance(record, SimFinCitation) else None

    def values_for(self, citation_id: str) -> list[float]:
        """Checkable evidence values behind a citation id, for the lints.

        Datapoint citations carry their own value; query-granularity
        citations expose every numeric cell of the recorded rows (the claim
        must appear somewhere in the actual result); document citations
        expose the numbers stated in the chunk text (unit-scaled, so
        "$1.4 billion" in a transcript yields 1.4 and 1.4e9). Charts carry no
        values.
        """
        record = self.citations.get(citation_id)
        if isinstance(record, SimFinCitation):
            if record.granularity == "datapoint":
                return [record.value] if record.value is not None else []
            values = [
                float(v)
                for row in self.sql_queries[record.sql_index].rows
                for v in row.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            return values[:500]
        if isinstance(record, DocumentCitation):
            if citation_id not in self._doc_values:
                from agents.citations import _UNIT_SCALE, parse_numbers

                values = []
                for number, unit, _ in parse_numbers(self._doc_texts.get(citation_id, "")):
                    values.append(number)
                    if unit in _UNIT_SCALE:
                        values.append(number * _UNIT_SCALE[unit])
                    elif unit == "%":
                        values.append(number / 100)
                self._doc_values[citation_id] = values[:500]
            return self._doc_values[citation_id]
        return []

    def cited_subset(self, ids: list[str]) -> list[Citation]:
        """The citation records for `ids`, deduplicated, in given order."""
        seen: set[str] = set()
        out: list[Citation] = []
        for i in ids:
            if i in self.citations and i not in seen:
                seen.add(i)
                out.append(self.citations[i])
        return out

    def _add(self, citation: Citation, *, key: tuple) -> Citation:
        existing = self._by_key.get(key)
        if existing is not None:
            return self.citations[existing]  # type: ignore[return-value]
        prefix = "S" if isinstance(citation, SimFinCitation) else "D"
        count = sum(1 for c in self.citations.values() if c.id.startswith(prefix))
        citation = citation.model_copy(update={"id": f"{prefix}{count + 1}"})
        self.citations[citation.id] = citation
        self._by_key[key] = citation.id
        return citation


def jsonable_row(row: dict[str, Any]) -> dict[str, Any]:
    """Coerce psycopg row values (Decimal, date, ...) to JSON-safe types.

    Must run BEFORE citation minting: psycopg returns NUMERIC as Decimal,
    which would fail the isinstance(int|float) metric check.
    """
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Decimal):
            out[key] = float(value)
        elif isinstance(value, (date, datetime)):
            out[key] = value.isoformat()
        else:
            out[key] = value
    return out


def _pick_str(row: dict[str, Any], names: tuple[str, ...]) -> str | None:
    lowered = {k.lower(): v for k, v in row.items()}
    for name in names:
        value = lowered.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _pick_int(row: dict[str, Any], names: tuple[str, ...]) -> int | None:
    lowered = {k.lower(): v for k, v in row.items()}
    for name in names:
        value = lowered.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None
