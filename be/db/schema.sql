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
    sector      TEXT,                 -- from SimFin industries dataset
    industry    TEXT,                 -- finer-grained than sector (e.g. 'Airlines')
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ticker)
);
-- sector/industry are backfilled by ingestion/enrich_companies.py (the
-- statement datasets don't carry them).

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

-- One row per source document (SEC filing or transcript).
CREATE TABLE IF NOT EXISTS documents (
    id               BIGSERIAL PRIMARY KEY,
    external_id      TEXT UNIQUE,     -- source-system (Octus) document id
    company_id       BIGINT REFERENCES companies (id),  -- optional SimFin link
    octus_company_id TEXT,
    company_name     TEXT,
    sub_industry     TEXT,
    source           TEXT NOT NULL,   -- 'SEC Filing' | 'Transcript'
    doc_type         TEXT,            -- '10-K' | '10-Q' | 'Transcript'
    title            TEXT,
    url              TEXT,
    document_date    DATE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- company_id is backfilled by ingestion/enrich_companies.py (normalized-name
-- matching of Octus company names onto SimFin companies); documents whose
-- company isn't in SimFin keep NULL and are reported by the backfill.

-- Chunked + embedded slices of documents for semantic search.
-- vector(512) matches the model2vec potion-retrieval-32M embedder; changing
-- the embedding model means changing this dimension AND re-indexing
-- (see core/config.py: EMBEDDING_MODEL / EMBEDDING_DIMENSION).
CREATE TABLE IF NOT EXISTS document_chunks (
    id            BIGSERIAL PRIMARY KEY,
    document_id   BIGINT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index   INT NOT NULL,
    content       TEXT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}',   -- citation + filter context
    embedding     vector(512),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- ANN index for cosine similarity search (pgvector HNSW).
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- Lexical leg of hybrid retrieval: full-text index over the chunk text.
-- What is indexed and why: `content` alone. Every chunk's content already
-- begins with its contextual header "[Company | doc type | date | section]"
-- and transcript chunks keep speaker attribution inline, so indexing content
-- covers body + company + doc type + section + speakers with one source of
-- truth — no drift between an indexed copy and the metadata.
-- Config 'english': stemming + stopword removal. Verified against this
-- corpus on PG16: hyphenated jargon indexes as compound AND parts
-- ('CASM-ex' -> 'casm-ex','casm','ex'), so websearch_to_tsquery phrase
-- queries match it; acronyms (TRASM, DOCSIS) and standard ids (ASC 606)
-- tokenize cleanly. No custom parser/dictionary until the retrieval
-- benchmark (evals/retrieval.py) demonstrates one is needed.
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_document_chunks_content_tsv
    ON document_chunks USING gin (content_tsv);

-- GIN index so JSONB containment filters (metadata @> ...) stay fast.
CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata
    ON document_chunks USING gin (metadata jsonb_path_ops);
