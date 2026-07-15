"""LangChain tool bindings for the multi-agent system.

Thin `@tool` wrappers over the existing, tested tool functions (schema_tool,
sql_tool, document_search_tool, chart_tool) — built PER REQUEST as closures
over that request's `EvidenceLedger`, so evidence capture needs no globals and
concurrent requests can't leak state into each other.

Two invariants enforced here:
- Wrappers never raise. Failures return "ERROR: ..." text, which LangChain
  delivers as an ordinary tool message the model can react to (same
  errors-are-feedback design as the requirement-2 loop).
- Everything citable is minted by the ledger at this boundary. SQL rows come
  back with their citation ids attached (`"cites": {"revenue": "S3"}`), and
  `create_chart` HYDRATES y-values from cited evidence — the model chooses
  which data points to plot but never retypes their numbers.
"""

from __future__ import annotations

import json
from typing import Any

from langchain.tools import tool
from pydantic import BaseModel

from agents.ledger import EvidenceLedger
from agents.orchestrator import MAX_SQL_ROWS, ROWS_SHOWN_TO_MODEL, SQL_TIMEOUT_MS
from tools import chart_tool, document_search_tool, schema_tool
from tools.sql_tool import run_read_only_sql

_MAX_SEARCH_K = 8


class ChartSeries(BaseModel):
    """One plotted series: a name plus the evidence ids to plot, in x order."""

    name: str
    citation_ids: list[str]


class TableRow(BaseModel):
    """One table row: its label plus one evidence id per value column."""

    label: str
    citation_ids: list[str]


def _fmt_value(value: float) -> str:
    """Human display for hydrated cells: 48211000000 -> '48.2B'."""
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= cutoff:
            return f"{value / cutoff:,.1f}{suffix}"
    if abs(value) >= 1e4:
        return f"{value / 1e3:,.1f}K"
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"


def make_toolsets(
    ledger: EvidenceLedger, database_url: str | None = None
) -> dict[str, list[Any]]:
    """Build the per-request tool lists for each specialist.

    Returns {"quant": [...], "docs": [...], "hybrid": quant ∪ docs}.
    """

    def _guard(fn, *args, **kwargs) -> str:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - errors are fed back to the model
            return f"ERROR: {type(exc).__name__}: {exc}"

    # -- discovery (shared with the requirement-2 loop, unchanged) -----------

    @tool
    def list_tables() -> str:
        """List every table in the financial database with approximate row
        counts and a one-line summary. Start here when unsure what exists."""
        return _guard(lambda: json.dumps(schema_tool.list_tables(database_url), default=str))

    @tool
    def describe_table(table: str) -> str:
        """Get a table's columns, curated usage notes (join keys, unit
        conventions, period-labeling quirks), and sample rows. Always describe
        a table before querying it for the first time."""
        return _guard(
            lambda: json.dumps(schema_tool.describe_table(table, database_url), default=str)
        )

    @tool
    def search_metrics(pattern: str) -> str:
        """Search the metric catalog of the long/tall `financials` table by
        case-insensitive substring. Metric names are data, not schema — always
        look them up here instead of guessing (there is NO 'eps' metric; EPS
        must be computed as net_income / shares_diluted)."""
        return _guard(
            lambda: json.dumps(schema_tool.search_metrics(pattern, database_url), default=str)
        )

    @tool
    def resolve_company(query: str) -> str:
        """Resolve a company name or ticker to its database id, canonical
        name, sector/industry, and data coverage. Always resolve before
        filtering by company. Returns up to 10 ranked candidates; pick by
        NAME, not rank alone (ticker 'FORD' is Forward Industries — Ford
        Motor is ticker 'F')."""
        return _guard(
            lambda: json.dumps(schema_tool.resolve_company(query, database_url), default=str)
        )

    # -- structured data ------------------------------------------------------

    @tool
    def query_financials(query: str) -> str:
        """Execute ONE read-only SELECT (or WITH … SELECT) against Postgres.
        Writes, DDL, and multiple statements are rejected; results are capped
        at 200 rows, so aggregate and LIMIT in SQL. CITATIONS: project ticker
        (or company name), fiscal_year, and fiscal_period columns — every
        numeric column of such rows is minted as a citable data point and
        returned under "cites" as {column: citation_id}. Cite those ids in
        your answer as [S1]. For cross-company comparisons, filter or group
        by currency — values are raw currency units, not all USD."""

        def run() -> str:
            rows = run_read_only_sql(
                query,
                max_rows=MAX_SQL_ROWS,
                statement_timeout_ms=SQL_TIMEOUT_MS,
                database_url=database_url,
            )
            safe_rows, cite_maps, nudge = ledger.record_sql(query, list(rows))
            shown = [
                {**row, "cites": cites}
                for row, cites in zip(safe_rows[:ROWS_SHOWN_TO_MODEL], cite_maps, strict=False)
            ]
            payload: dict[str, Any] = {"row_count": len(safe_rows), "rows": shown}
            if len(safe_rows) > ROWS_SHOWN_TO_MODEL:
                payload["note"] = (
                    f"Showing first {ROWS_SHOWN_TO_MODEL} of {len(safe_rows)} rows — "
                    "aggregate or LIMIT in SQL to narrow the result."
                )
            if len(safe_rows) == MAX_SQL_ROWS:
                payload["note"] = (
                    f"Row cap of {MAX_SQL_ROWS} hit; the true result may be larger. "
                    "Aggregate in SQL instead of enumerating rows."
                )
            if nudge:
                payload["citation_note"] = nudge
            return json.dumps(payload, default=str)

        return _guard(run)

    # -- documents --------------------------------------------------------------

    @tool
    def document_coverage() -> str:
        """What the document corpus covers: which companies (with tickers),
        document types (10-K/10-Q/Transcript), sub-industries, date ranges,
        allowed search filter keys, transcript qa_role values, and the most
        common filing sections. The corpus holds ~12 companies — call this
        FIRST to scope any sector/trend question to real coverage."""
        return _guard(
            lambda: json.dumps(
                document_search_tool.document_coverage(database_url), default=str
            )
        )

    @tool
    def search_documents(
        query: str,
        k: int = 5,
        company_name: str | None = None,
        sub_industry: str | None = None,
        doc_type: str | None = None,
        section: str | None = None,
        content_kind: str | None = None,
        qa_role: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> str:
        """Search over SEC filings and earnings-call transcripts (the
        retrieval strategy — dense, lexical, or hybrid — is server
        configuration, not a parameter).
        Returns the most relevant chunks, each with a minted citation id —
        cite them as [D1]. Use exact filter values from document_coverage
        (e.g. doc_type='10-K', sub_industry='Passenger Airlines',
        qa_role='Answer' for management's transcript answers); dates are ISO
        strings bounding the document date."""

        def run() -> str:
            filters = {
                key: value
                for key, value in {
                    "company_name": company_name,
                    "sub_industry": sub_industry,
                    "doc_type": doc_type,
                    "section": section,
                    "content_kind": content_kind,
                    "qa_role": qa_role,
                    "date_from": date_from,
                    "date_to": date_to,
                }.items()
                if value
            }
            results = document_search_tool.search_documents(
                query,
                k=min(int(k), _MAX_SEARCH_K),
                filters=filters or None,
                database_url=database_url,
            )
            citations = ledger.record_documents(query, filters, results)
            payload = [
                {
                    "cite": c.id,
                    "title": r.title,
                    "doc_type": c.doc_type,
                    "company": c.company_name,
                    "section": c.section,
                    "date": c.document_date,
                    "score": c.score,
                    "text": r.text,
                }
                for r, c in zip(results, citations, strict=True)
            ]
            return json.dumps({"result_count": len(payload), "chunks": payload}, default=str)

        return _guard(run)

    # -- charts -----------------------------------------------------------------

    @tool
    def create_chart(
        title: str,
        kind: str,
        x: list[str],
        series: list[ChartSeries],
        y_label: str | None = None,
    ) -> str:
        """Render a line or bar chart from ALREADY-CITED data points. Pass x
        labels (e.g. ["Q1 2025", "Q2 2025"]) and, per series, the SimFin
        citation ids (e.g. ["S3", "S5"]) aligned with x — one id per label.
        The y-values are looked up from the cited evidence; you never pass
        numbers. Returns the chart id and image URL — reference the chart in
        your answer as a markdown image: ![title](url)."""

        def run() -> str:
            hydrated = []
            used_ids: list[str] = []
            for s in series:
                if len(s.citation_ids) != len(x):
                    raise ValueError(
                        f"Series {s.name!r} has {len(s.citation_ids)} citation ids "
                        f"but x has {len(x)} labels — they must align."
                    )
                if len(set(s.citation_ids)) != len(s.citation_ids):
                    raise ValueError(
                        f"Series {s.name!r} repeats a citation id — each x "
                        "position must plot its own cited data point."
                    )
                values = []
                for cid, label in zip(s.citation_ids, x, strict=True):
                    record = ledger.get_simfin(cid)
                    if record is None or record.granularity != "datapoint":
                        raise ValueError(
                            f"{cid!r} is not a SimFin data-point citation id; "
                            "only cited data points can be plotted."
                        )
                    if record.fiscal_period and record.fiscal_year:
                        expected = (record.fiscal_period, str(record.fiscal_year))
                        if (
                            record.fiscal_period not in label.upper()
                            or str(record.fiscal_year) not in label
                        ):
                            raise ValueError(
                                f"x label {label!r} does not name the cited "
                                f"point's period {expected[0]} {expected[1]} — "
                                "label each x position with its fiscal period "
                                "and year (e.g. 'Q1 2025')."
                            )
                    values.append(record.value)
                    used_ids.append(cid)
                hydrated.append({"name": s.name, "values": values})

            rendered = chart_tool.render_chart(
                {"title": title, "kind": kind, "x": x, "series": hydrated, "y_label": y_label}
            )
            chart = ledger.record_chart(rendered, used_ids)
            return json.dumps(
                {
                    "chart_id": chart.id,
                    "url": chart.url,
                    "markdown": f"![{title}]({chart.url})",
                    "plotted": hydrated,
                }
            )

        return _guard(run)

    @tool
    def create_table(title: str, columns: list[str], rows: list[TableRow]) -> str:
        """Build a structured multi-metric table from ALREADY-CITED data
        points. columns: header names — the FIRST is the row-label column
        (e.g. 'Quarter'), the rest are value columns (e.g. 'Revenue',
        'Net Income', 'EPS'). Each row gives its label plus one SimFin
        citation id per value column; the values are looked up from the
        cited evidence — you never pass numbers. Returns ready-to-embed
        markdown (already carrying the citation markers): paste it into
        your answer verbatim."""

        def run() -> str:
            from agents.schemas import TableCell

            if len(columns) < 2:
                raise ValueError(
                    "Need at least a label column and one value column."
                )
            cell_rows: list[list[TableCell]] = []
            lines = [
                "| " + " | ".join(columns) + " |",
                "|" + "---|" * len(columns),
            ]
            for row in rows:
                if len(row.citation_ids) != len(columns) - 1:
                    raise ValueError(
                        f"Row {row.label!r} has {len(row.citation_ids)} citation "
                        f"ids but there are {len(columns) - 1} value columns."
                    )
                cells = [TableCell(text=row.label)]
                rendered = [row.label]
                for cid in row.citation_ids:
                    record = ledger.get_simfin(cid)
                    if record is None or record.granularity != "datapoint":
                        raise ValueError(
                            f"{cid!r} is not a SimFin data-point citation id; "
                            "only cited data points can fill table cells."
                        )
                    cells.append(TableCell(value=record.value, citation_id=cid))
                    rendered.append(f"{_fmt_value(record.value)} [{cid}]")
                cell_rows.append(cells)
                lines.append("| " + " | ".join(rendered) + " |")

            markdown = "\n".join(lines)
            table = ledger.record_table(title, columns, cell_rows, markdown)
            return json.dumps({"table_id": table.id, "markdown": markdown})

        return _guard(run)

    quant = [
        list_tables,
        describe_table,
        search_metrics,
        resolve_company,
        query_financials,
        create_chart,
        create_table,
    ]
    docs = [document_coverage, search_documents, resolve_company]
    hybrid = quant + [document_coverage, search_documents]
    return {"quant": quant, "docs": docs, "hybrid": hybrid}
