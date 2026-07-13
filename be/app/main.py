"""FastAPI entrypoint for the Financial Multi-Agent Intelligence System.

Exposes the agent layer to clients:
- GET  /health — liveness probe.
- POST /ask    — natural-language question -> traceable answer (prose + the
                 SQL queries and tool calls that produced it).

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.schemas import AgentAnswer

app = FastAPI(
    title="Financial Multi-Agent Intelligence System",
    version="0.1.0",
    description=(
        "Backend agent system that answers financial questions with citations "
        "and charts, backed by SimFin structured data and SEC/earnings documents."
    ),
)


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


# TODO: Extend /ask to orchestrate document search + chart tools alongside SQL
#       (later case-study requirements) and stream progress to the frontend.
