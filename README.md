# Financial Multi-Agent Intelligence System

A case-study system that answers financial questions with **citations and charts**.
It combines two kinds of knowledge about companies:

- **Structured** financial data (SimFin) stored in Postgres, queried with SQL.
- **Unstructured** text (SEC filings, earnings-call transcripts) chunked, embedded,
  and stored in a **pgvector** index for semantic search.

A backend **agent** interprets a natural-language question, decides which tools to
use (SQL, document search, charting), and returns a grounded answer with the
sources it relied on.

> Status: **scaffold**. The structure, configs, and typed placeholders are in
> place; core functionality is intentionally not implemented yet.

## Repository layout

| Path | What it is |
| --- | --- |
| [`/be`](./be) | Python **backend**: FastAPI API, agent orchestration, ingestion, retrieval, tools, and the database schema. |
| [`/fe`](./fe) | **Frontend** placeholder: chat UI that renders answers, citations, and charts. |
| [`/infra`](./infra) | **Infrastructure**: Docker Compose for local Postgres + pgvector. |

## Architecture

```
                          ┌──────────────────────────────────────────┐
   SimFin API ──ingest──► │ Postgres      companies, financials       │
                          │  + pgvector   documents, document_chunks  │
 SEC / earnings ─ingest─► └──────────────────────────────────────────┘
   documents                         ▲            ▲
                                     │ SQL        │ vector search
                                     │            │
   user question ─► FastAPI ─► Agent ─► Tools ────┴────────────
                    (/be/app)  (/be/agents)  (sql | document_search | chart)
                                     │
                                     ▼
                    answer + citations + chart specs ─► Frontend (/fe)
```

- **Ingestion** (`be/ingestion`) loads SimFin into Postgres and chunks/embeds
  documents into pgvector.
- **Tools** (`be/tools`) expose safe, single-purpose capabilities: read-only SQL,
  semantic document search, and chart-spec generation.
- **Agent** (`be/agents`) orchestrates the tools and composes a cited answer.
- **API** (`be/app`) serves it over FastAPI to the frontend.

## Local development quickstart

Prerequisites: **Docker** (for Postgres) and **Python 3.11+**.

```bash
# 1. Start local infrastructure (Postgres + pgvector).
cd infra
cp .env.example .env
docker compose up -d
cd ..

# 2. Set up the backend.
cd be
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # ensure DATABASE_URL matches infra/.env

# 3. Apply the database schema.
psql "$DATABASE_URL" -f db/schema.sql

# 4. Run the API.
uvicorn app.main:app --reload --port 8000
# -> http://localhost:8000/health
```

See [`be/README.md`](./be/README.md) and [`infra/README.md`](./infra/README.md) for
details.

## From local to production

Local development uses **Docker Compose**; the design maps cleanly onto managed AWS
services when it is time to deploy:

| Local (Docker Compose) | Production (AWS) |
| --- | --- |
| Backend container | **ECS / Fargate** |
| Postgres + pgvector | **RDS / Aurora Postgres** with pgvector |
| Local file storage | **S3** (raw filings, transcripts, caches) |
| `.env` files | **Secrets Manager** / SSM Parameter Store |
| Container logs | **CloudWatch** + **OpenTelemetry** (traces/metrics) |
