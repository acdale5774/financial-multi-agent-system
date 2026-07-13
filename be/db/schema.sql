-- Schema for the Financial Multi-Agent Intelligence System.
--
-- Two data domains live side by side in Postgres:
--   1. Structured SimFin financials (companies, financials).
--   2. Unstructured documents chunked + embedded for semantic search
--      (documents, document_chunks) using the pgvector extension.
--
-- Apply locally with:
--   psql "$DATABASE_URL" -f be/db/schema.sql
--
-- NOTE: This is an initial sketch. Columns and indexes will evolve as the
--       ingestion normalization rules are finalized.

-- pgvector: required for the `vector` column type used by document_chunks.
CREATE EXTENSION IF NOT EXISTS vector;


-- --------------------------------------------------------------------------
-- Structured financial data (SimFin)
-- --------------------------------------------------------------------------

-- One row per company we track.
CREATE TABLE IF NOT EXISTS companies (
    id          BIGSERIAL PRIMARY KEY,
    ticker      TEXT NOT NULL,
    name        TEXT NOT NULL,
    simfin_id   TEXT UNIQUE,          -- stable SimFin identifier
    sector      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ticker)
);

-- Normalized financial line items. Kept intentionally generic (long/tall shape)
-- so income/balance/cashflow metrics can share one table.
-- TODO: Revisit once normalize_simfin_data() defines the canonical metric set;
--       consider a wide, per-statement layout if that reads more clearly.
CREATE TABLE IF NOT EXISTS financials (
    id            BIGSERIAL PRIMARY KEY,
    company_id    BIGINT NOT NULL REFERENCES companies (id),
    statement     TEXT NOT NULL,      -- 'income' | 'balance' | 'cashflow'
    fiscal_year   INT NOT NULL,
    fiscal_period TEXT NOT NULL,      -- 'FY' (annual) | 'Q1'..'Q4' (quarterly)
    currency      TEXT,
    report_date   DATE,               -- period end / report date from SimFin
    metric        TEXT NOT NULL,      -- snake_cased, e.g. 'revenue', 'net_income'
    value         NUMERIC,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (company_id, statement, fiscal_year, fiscal_period, metric)
);

CREATE INDEX IF NOT EXISTS idx_financials_company_period
    ON financials (company_id, fiscal_year, fiscal_period);


-- --------------------------------------------------------------------------
-- Unstructured documents (SEC filings, earnings-call transcripts)
-- --------------------------------------------------------------------------

-- One row per source document.
CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    company_id    BIGINT REFERENCES companies (id),
    source        TEXT NOT NULL,      -- 'sec_filing' | 'earnings_transcript'
    doc_type      TEXT,               -- e.g. '10-K', '10-Q', 'earnings_call'
    title         TEXT,
    url           TEXT,
    published_at  DATE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Chunked + embedded slices of documents for semantic search.
-- The embedding dimension MUST match the embedding model chosen at ingest time.
-- TODO: Set the real dimension (e.g. 1536) once the embedding model is chosen.
CREATE TABLE IF NOT EXISTS document_chunks (
    id            BIGSERIAL PRIMARY KEY,
    document_id   BIGINT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index   INT NOT NULL,
    content       TEXT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}',   -- citation context
    embedding     vector(1536),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- TODO: Add an ANN index for fast similarity search once data exists and the
--       distance metric is chosen, e.g.:
--   CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);
--       (ivfflat is an alternative; it needs ANALYZE + a `lists` tuning value.)
