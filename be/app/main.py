"""FastAPI entrypoint for the Financial Multi-Agent Intelligence System.

This is the API layer that will eventually expose the agent system to clients.
For now it only provides a health check so the service can be started and wired
into local infrastructure (Docker Compose) and future deployment targets.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="Financial Multi-Agent Intelligence System",
    version="0.1.0",
    description=(
        "Backend agent system that answers financial questions with citations "
        "and charts, backed by SimFin structured data and SEC/earnings documents."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe used by Docker Compose, load balancers, and monitoring."""
    return {"status": "ok"}


# TODO: Add a POST /ask endpoint that accepts a natural-language financial
#       question and routes it through the agent system (see /agents), returning
#       an answer with citations and optional chart specifications.
# TODO: Wire application settings (DB URL, embedding provider, model config)
#       via pydantic-settings loading from the environment (.env.example).
# TODO: Add startup/shutdown lifecycle hooks to manage the Postgres connection
#       pool and any long-lived agent/tool resources.
