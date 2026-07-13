"""FastAPI entrypoint for the Financial Multi-Agent Intelligence System.

Exposes the agent layer to clients:
- GET  /health    — liveness probe.
- POST /agent/ask — the multi-agent system (requirement 3): routed
                    specialists, cited answer, charts, full evidence trace.
- POST /ask       — the single-agent structured-data interface
                    (requirement 2), kept untouched as a baseline/fallback.
- GET  /charts/*  — rendered chart PNGs referenced by answers.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.schemas import AgentAnswer, MultiAgentAnswer
from tools.chart_tool import CHART_DIR

app = FastAPI(
    title="Financial Multi-Agent Intelligence System",
    version="0.1.0",
    description=(
        "Backend agent system that answers financial questions with citations "
        "and charts, backed by SimFin structured data and SEC/earnings documents."
    ),
)

CHART_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/charts", StaticFiles(directory=str(CHART_DIR)), name="charts")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe used by Docker Compose, load balancers, and monitoring."""
    return {"status": "ok"}


@app.post("/ask", response_model=AgentAnswer)
def ask(request: AskRequest) -> AgentAnswer:
    """Answer a financial question over the structured store, with audit trail."""
    # Imported here so the service starts (and /health works) even when the
    # openai dependency or API key isn't configured yet.
    import openai

    from agents.orchestrator import answer

    try:
        return answer(request.question)
    except openai.AuthenticationError as exc:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key missing or invalid — set OPENAI_API_KEY in be/.env.",
        ) from exc
    except openai.APIStatusError as exc:
        raise HTTPException(
            status_code=502, detail=f"Upstream model error ({exc.status_code}): {exc.message}"
        ) from exc
    except openai.APIConnectionError as exc:
        raise HTTPException(status_code=502, detail="Could not reach the OpenAI API.") from exc


@app.post("/agent/ask", response_model=MultiAgentAnswer)
def agent_ask(request: AskRequest) -> MultiAgentAnswer:
    """Answer via the multi-agent system: routed, cited, chart-capable."""
    import openai

    from agents.multi_agent import answer_multi

    try:
        return answer_multi(request.question)
    except openai.AuthenticationError as exc:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key missing or invalid — set OPENAI_API_KEY in be/.env.",
        ) from exc
    except openai.APIStatusError as exc:
        raise HTTPException(
            status_code=502, detail=f"Upstream model error ({exc.status_code}): {exc.message}"
        ) from exc
    except openai.APIConnectionError as exc:
        raise HTTPException(status_code=502, detail="Could not reach the OpenAI API.") from exc
