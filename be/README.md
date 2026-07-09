# Backend (`/be`)

Python backend for the Financial Multi-Agent Intelligence System. It ingests data,
stores it in Postgres (structured + pgvector), and exposes a FastAPI agent system
that answers financial questions with **citations and charts**.

> Status: **scaffold**. Modules are typed placeholders with docstrings and TODOs —
> no full functionality yet.

## Layout

```
be/
├── app/          FastAPI entrypoint (API layer)
├── agents/       Agent orchestration — picks and calls tools, composes answers
├── ingestion/    SimFin -> Postgres; documents -> chunk/embed/index
├── retrieval/    Structured (SQL) + semantic (vector) retrieval over stored data
├── tools/        Agent-callable tools: sql / document_search / chart
├── db/           schema.sql (Postgres + pgvector) and notes
└── tests/        pytest suite
```

## Data flow

```
SimFin API ──► ingestion.simfin_ingest ──► Postgres (companies, financials)
Documents  ──► ingestion.document_ingest ─► pgvector (document_chunks)

question ─► agents ─► tools ─► [ sql_tool | document_search_tool | chart_tool ]
                                        │
                                        ▼
                          answer + citations + chart specs  ─► FastAPI (app) ─► FE
```

## Local development

Prerequisites: Python 3.11+, and local Postgres running (see
[`../infra`](../infra)).

```bash
cd be

# 1. Create a virtual environment and install (editable) with dev extras.
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure environment.
cp .env.example .env        # then fill in values

# 3. Apply the database schema (requires Postgres from ../infra).
psql "$DATABASE_URL" -f db/schema.sql

# 4. Run the API.
uvicorn app.main:app --reload --port 8000
# -> http://localhost:8000/health
```

## Tooling

```bash
ruff check .     # lint
mypy .           # type-check
pytest           # tests
```

## Configuration

All configuration comes from the environment — see [`.env.example`](./.env.example).
Key variable: `DATABASE_URL` (Postgres connection string).
