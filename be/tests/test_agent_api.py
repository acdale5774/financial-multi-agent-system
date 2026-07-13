"""API contract tests for the multi-agent endpoint and chart serving."""

from __future__ import annotations

from fastapi.testclient import TestClient

from agents import multi_agent
from agents.schemas import MultiAgentAnswer, ValidationReport
from app.main import app
from tools.chart_tool import CHART_DIR


def test_agent_ask_returns_cited_answer(monkeypatch):
    def fake_answer(question):
        return MultiAgentAnswer(
            question=question,
            route="quant",
            route_reason="numbers",
            answer="Revenue was $185.0B [S1].",
            citations=[],
            charts=[],
            validation=ValidationReport(markers_found=1),
            sql_queries=[],
            searches=[],
            model="gpt-5.5",
        )

    monkeypatch.setattr(multi_agent, "answer_multi", fake_answer)
    client = TestClient(app)

    response = client.post("/agent/ask", json={"question": "Ford revenue?"})

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "quant"
    assert body["validation"]["markers_found"] == 1
    assert "[S1]" in body["answer"]


def test_chart_static_mount_serves_pngs():
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    target = CHART_DIR / "test-chart.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        client = TestClient(app)
        response = client.get("/charts/test-chart.png")
        assert response.status_code == 200
        assert response.content.startswith(b"\x89PNG")
    finally:
        target.unlink(missing_ok=True)
