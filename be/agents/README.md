# Agents

The agent layer is the "brain" of the system. It receives a natural-language
financial question and decides how to answer it by orchestrating the tools in
[`../tools`](../tools).

## Responsibilities

- Interpret the user's financial question.
- Decide which tool(s) to call and in what order:
  - **SQL tool** — query structured SimFin financials in Postgres.
  - **Document search tool** — retrieve relevant passages from SEC filings /
    earnings call transcripts via the vector index.
  - **Chart tool** — turn structured results into chart specifications.
- Compose a final answer that **cites its sources** (SQL rows and/or document
  chunks) and optionally attaches one or more charts.

## Design intent

- Keep the orchestration logic thin and explicit — easy to trace what the agent
  did and why (important for a financial/auditable context).
- Tools are plain, independently testable functions; the agent is the only piece
  that decides *when* to call them.

## Planned structure (not implemented yet)

- `orchestrator.py` — top-level entrypoint: `answer(question) -> Answer`.
- `prompts.py` — system/tool prompts and answer-formatting templates.
- `schemas.py` — pydantic models for questions, tool calls, and cited answers.

> TODO: Choose the agent framework/approach (native tool-calling loop vs. a
> higher-level framework) and document the decision here.
