"""Tests for the deterministic chart renderer (tools/chart_tool.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.chart_tool import render_chart


def _spec(**overrides):
    spec = {
        "title": "Revenue by quarter",
        "kind": "line",
        "x": ["2024 Q1", "2024 Q2", "2024 Q3"],
        "series": [{"name": "F", "values": [1.0, 2.0, None]}],
        "y_label": "USD",
    }
    spec.update(overrides)
    return spec


def test_renders_line_chart_png(tmp_path):
    out = render_chart(_spec(), output_dir=tmp_path)

    png = Path(out["file"])
    assert png.exists() and png.read_bytes()[:4] == b"\x89PNG"
    assert out["url"] == f"/charts/{png.name}"
    assert out["spec"]["title"] == "Revenue by quarter"


def test_renders_multi_series_bar_chart(tmp_path):
    spec = _spec(
        kind="bar",
        series=[
            {"name": "F", "values": [1.0, 2.0, 3.0]},
            {"name": "DAL", "values": [2.0, 1.0, 2.5]},
        ],
    )
    out = render_chart(spec, output_dir=tmp_path)
    assert Path(out["file"]).stat().st_size > 0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"kind": "pie"}, "must be one of"),
        ({"title": ""}, "title"),
        ({"x": []}, "'x' must be"),
        ({"series": []}, "'series' must be"),
        ({"series": [{"name": "F", "values": [1.0]}]}, "must align"),
        ({"series": [{"name": "F", "values": ["a", "b", "c"]}]}, "must be numbers"),
    ],
)
def test_malformed_specs_raise_value_error(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        render_chart(_spec(**overrides), output_dir=tmp_path)
