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

## Data ingestion (SimFin → Postgres)

Structured financials are pulled with the [`simfin`](https://github.com/SimFin/simfin)
SDK (bulk datasets), normalized to a **long/tall** shape (one row per
company/statement/period/metric), and upserted into Postgres. Set a real
`SIMFIN_API_KEY` in `.env` (a free account at https://app.simfin.com/ gives you a
key — the legacy `free` key now returns HTTP 401 on bulk downloads).

```bash
# Annual income/balance/cashflow; --init-db applies db/schema.sql first.
python -m ingestion.simfin_ingest --init-db --variant annual

# Quarterly (Q1–Q4 by fiscal year):
python -m ingestion.simfin_ingest --variant quarterly --statements income,balance,cashflow
```

Re-running is idempotent (`ON CONFLICT … DO UPDATE`) — it refreshes values rather
than duplicating rows. Once loaded, data is retrievable via the read-only SQL tool
(`tools/sql_tool.py`), e.g.:

```python
from tools.sql_tool import run_read_only_sql

run_read_only_sql(
    "SELECT c.ticker, f.fiscal_year, f.value "
    "FROM financials f JOIN companies c ON c.id = f.company_id "
    "WHERE f.metric = 'revenue' AND f.statement = 'income' "
    "ORDER BY f.fiscal_year"
)
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
