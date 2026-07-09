"""Turn structured query results into a chart specification.

The tool returns a **declarative chart spec** (a plain dict, Vega-Lite-shaped)
rather than a rendered image. This keeps the backend free of heavy plotting
dependencies and lets the frontend render interactively; a server-side renderer
can be added later if static images are needed.
"""

from __future__ import annotations

from typing import Any, Literal

ChartType = Literal["line", "bar", "area", "scatter"]

# A declarative chart specification consumed by the frontend renderer.
ChartSpec = dict[str, Any]


def build_chart(
    rows: list[dict[str, Any]],
    *,
    chart_type: ChartType,
    x: str,
    y: str,
    series: str | None = None,
    title: str | None = None,
) -> ChartSpec:
    """Build a declarative chart spec from tabular data.

    Args:
        rows: Tabular data, typically the output of the SQL tool.
        chart_type: Kind of chart to produce.
        x: Field name for the x-axis (e.g. "fiscal_year").
        y: Field name for the y-axis (e.g. "revenue").
        series: Optional field to split the data into multiple series
            (e.g. "company" for a multi-company comparison).
        title: Optional chart title.

    Returns:
        A chart specification dict the frontend can render.

    TODO: Emit a valid Vega-Lite spec (encode `x`, `y`, optional color=`series`),
          infer sensible axis types, and validate that the referenced fields
          exist in `rows`.
    """
    raise NotImplementedError
