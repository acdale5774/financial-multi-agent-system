"""Ingestion package.

Two ingestion paths feed the system:

- `simfin_ingest`  — structured financial data (SimFin) -> Postgres tables.
- `document_ingest` — unstructured text (SEC filings, earnings call transcripts)
  -> chunked, embedded, and stored in the pgvector index.
"""
