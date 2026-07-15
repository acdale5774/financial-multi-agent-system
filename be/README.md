# Backend (`/be`)

Python backend for the Financial Multi-Agent Intelligence System. It ingests data,
stores it in Postgres (structured + pgvector), and exposes a FastAPI agent system
that answers financial questions with **citations and charts**.

> Status: ingestion (structured + documents), the single-agent query interface
> (`POST /ask`), and the multi-agent system (`POST /agent/ask`, routed
> specialists with per-claim citations and charts) are implemented and tested.
> See [`agents/README.md`](./agents/README.md) for the agent architecture.

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

After loading, enrich the store (idempotent; fulfills two schema TODOs):

```bash
# companies.sector/industry from SimFin's industries taxonomy, and
# documents.company_id links via normalized-name matching (cross-source joins)
python -m ingestion.enrich_companies
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

## Document ingestion (SEC filings + transcripts → pgvector)

The `case_study_data/` corpus (137 SEC filings as Workiva HTML, 105 transcripts
as HTML-in-JSON) is chunked, embedded, and indexed into the pgvector
`document_chunks` table:

```bash
python -m ingestion.document_ingest --init-db          # full corpus
python -m ingestion.document_ingest --limit 5          # smoke test
```

### Chunking strategy (and why)

Financial/legal text has strong native structure; the chunker follows it
instead of using fixed-size windows:

| Choice | Rationale |
| --- | --- |
| **Section-aware (filings)** — `Item 1A`, `Item 7`… headers are hard chunk boundaries; every chunk carries its section | 10-K/10-Q Items are self-contained legal units. A window spanning Risk Factors → MD&A produces chunks that embed as neither. Section metadata also enables scoped retrieval ("search only Risk Factors"). |
| **Table-atomic** — financial tables are linearized row-by-row (`Revenue \| 61,643 \| 58,048`) and never split mid-table; flagged `content_kind='table'` | Splitting a table separates labels from values, destroying its meaning. Row linearization keeps each figure attached to its label and period. Workiva *layout* tables (cover pages) are detected by shape and flattened to plain text instead. |
| **Turn-aware (transcripts)** — the unit is the speaker turn; Q&A pairs pack together within a section; attribution kept inline and in metadata | "Who said it" is the signal in a transcript — a CFO's answer ≠ an analyst's question. Splitting mid-turn orphans the statement from its speaker; merging across the Management-Discussion/Q&A boundary blurs prepared remarks with spontaneous answers. |
| **Contextual header** — every chunk is prefixed `[Company \| doc type \| date \| section]` | A lightweight version of contextual retrieval: boilerplate passages ("fuel costs increased…") embed with their document identity, and citations render directly from the chunk. |
| **~2,000-char target / 3,000 max, semantic boundaries** | Big enough for a complete thought or table, small enough for precise retrieval. Overlap is used only when hard-splitting an oversized unit (200 chars) — semantic boundaries make blind sliding-window overlap unnecessary. |

### Vector database choice: pgvector

Postgres + pgvector over Pinecone/FAISS, deliberately:

- **One store, one join key** — structured SimFin financials and document chunks
  live in the same database, so "retrievable alongside" is a SQL join, not a
  cross-system sync. One `docker compose up`, one backup, one prod target
  (RDS/Aurora supports pgvector).
- **Real filtered retrieval** — metadata filters are indexed SQL (`JSONB @>`
  with a GIN index) combined with HNSW ANN search in one query — no
  application-side post-filtering like FAISS, no separate metadata-sync
  pipeline like Pinecone.
- **Right scale** — ~40k chunks × 512 dims is far below where a dedicated
  vector DB pays for its operational cost. FAISS is an in-process index
  (persistence/filtering DIY); Pinecone is a managed service (network hop,
  another vendor) — justified at 10-100M+ vectors, not here.

### Retrieval strategies: dense / lexical / hybrid

Document search is no longer dense-only. A generated `tsvector` column (GIN
index) adds Postgres full-text retrieval, and `RETRIEVAL_STRATEGY` selects
`dense`, `lexical`, or `hybrid` (both legs fused with Reciprocal Rank
Fusion). The default is **hybrid**, chosen from a 29-case labelled benchmark
(`python -m evals.retrieval`; hybrid 77% Recall@10 / 0.655 MRR vs dense-only
65% / 0.481). Design, trade-offs, and the honest limitations are in
[`retrieval/README.md`](./retrieval/README.md); the committed scorecard is
[`evals/retrieval_report.md`](./evals/retrieval_report.md).

### Embeddings

Local **model2vec** static embeddings (`potion-retrieval-32M`, 512-dim): no API
key, free, offline, and the only local option that installs on this dev
machine (Intel Mac + Python 3.14 — torch/onnxruntime have no wheels).
Trade-off: below transformer-quality retrieval, mitigated by structure-aware
chunks + contextual headers, and now measured by the retrieval benchmark
(dense-only is the weakest strategy — a local-dev constraint, not a
production recommendation). The `Embedder` protocol
(`ingestion/embeddings.py`) makes upgrading (e.g. Voyage `voyage-finance-2`)
a config change + schema dimension bump + re-index.

### Chunk metadata schema

Every chunk carries JSONB metadata for precise filtered retrieval
(`tools/document_search_tool.py`):

| Key | Example | Purpose |
| --- | --- | --- |
| `company_name` / `octus_company_id` / `sub_industry` | `Delta Air Lines` / `2483` / `Passenger Airlines` | Scope to a company or industry |
| `source_type` / `doc_type` | `SEC Filing` / `10-K` | Filings vs transcripts; form type |
| `document_date` | `2024-10-10` | Time-bounded queries (`date_from`/`date_to`) |
| `section` | `Part I · Item 1A. Risk Factors`, `Q&A` | Scope to a filing Item or call section |
| `content_kind` | `text` \| `table` | Prefer tables for numeric questions |
| `speakers` / `qa_role` | `["Glen Hauenstein"]` / `Answer` | Transcript attribution filters |

```python
from tools.document_search_tool import search_documents

search_documents(
    "fuel cost outlook",
    k=5,
    filters={"company_name": "Delta Air Lines", "doc_type": "10-Q"},
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
