"""Deterministic chart rendering for agent answers.

The agent supplies a small declarative spec (title, kind, categorical x values,
one or more named series); this module renders it to a PNG under `_charts/`
and returns both the file location and the spec. The spec travels in the API
response so a future frontend can re-render natively, while the PNG makes the
answer viewable with nothing but curl and an image viewer.

Rendering is plain code, not an LLM step: chart correctness is a data-fidelity
concern, and the numbers must come verbatim from cited evidence.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

# Where PNGs land, relative to the backend working directory; served by the
# FastAPI app at /charts. Gitignored.
CHART_DIR = Path("_charts")

_ALLOWED_KINDS = ("line", "bar")


def render_chart(spec: dict[str, Any], *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Render a validated chart spec to a PNG file.

    Args:
        spec: {"title": str, "kind": "line"|"bar", "x": [str, ...],
               "series": [{"name": str, "values": [number|None, ...]}, ...],
               "y_label": str (optional)}.
              Each series must have exactly len(x) values; None marks a gap.
        output_dir: Override the chart directory (tests use tmp_path).

    Returns:
        {"file": absolute PNG path, "url": served path, "spec": the spec}.

    Raises:
        ValueError: on a malformed spec (fed back to the model to fix).
    """
    title = spec.get("title")
    kind = spec.get("kind")
    x = spec.get("x")
    series = spec.get("series")

    if not title or not isinstance(title, str):
        raise ValueError("Chart spec needs a non-empty string 'title'.")
    if kind not in _ALLOWED_KINDS:
        raise ValueError(f"Chart 'kind' must be one of {_ALLOWED_KINDS}, got {kind!r}.")
    if not isinstance(x, list) or not x or not all(isinstance(v, str) for v in x):
        raise ValueError("'x' must be a non-empty list of category labels (strings).")
    if not isinstance(series, list) or not series:
        raise ValueError("'series' must be a non-empty list.")
    for s in series:
        if not isinstance(s, dict) or not s.get("name") or not isinstance(s.get("values"), list):
            raise ValueError("Each series needs a 'name' and a 'values' list.")
        if len(s["values"]) != len(x):
            raise ValueError(
                f"Series {s.get('name')!r} has {len(s['values'])} values "
                f"but 'x' has {len(x)} labels — they must align."
            )
        if not all(v is None or isinstance(v, (int, float)) for v in s["values"]):
            raise ValueError(f"Series {s.get('name')!r} values must be numbers (or null for gaps).")

    # Figure + Agg canvas directly (not pyplot): pyplot keeps global figure
    # state that is unsafe under concurrent FastAPI requests, and Agg needs
    # no display server.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter

    fig = Figure(figsize=(8, 4.5), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    positions = range(len(x))
    if kind == "line":
        for s in series:
            ax.plot(positions, [_none_to_nan(v) for v in s["values"]], marker="o", label=s["name"])
    else:
        width = 0.8 / len(series)
        for i, s in enumerate(series):
            offsets = [p + (i - (len(series) - 1) / 2) * width for p in positions]
            ax.bar(offsets, [_none_to_nan(v) for v in s["values"]], width=width, label=s["name"])

    ax.set_title(title)
    ax.set_xticks(list(positions), x, rotation=30, ha="right")
    if spec.get("y_label"):
        ax.set_ylabel(str(spec["y_label"]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: _human_number(v)))
    ax.grid(True, axis="y", alpha=0.3)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()

    out_dir = Path(output_dir) if output_dir else CHART_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = (out_dir / f"chart-{uuid.uuid4().hex[:12]}.png").resolve()
    fig.savefig(path)

    return {"file": str(path), "url": f"/charts/{path.name}", "spec": spec}


def _none_to_nan(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _human_number(value: float) -> str:
    """1234567890 -> '1.2B' so raw-currency axes stay readable."""
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= cutoff:
            return f"{value / cutoff:.1f}{suffix}"
    return f"{value:g}"
