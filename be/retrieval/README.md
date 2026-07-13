# Retrieval

The retrieval layer sits between stored data and the agent. It answers the
question *"what information is relevant to this query?"* from two complementary
sources:

| Source | Store | Access pattern | Used for |
| --- | --- | --- | --- |
| SimFin financials | Postgres tables | Read-only SQL | Precise numbers, ratios, time series |
| SEC filings / transcripts | pgvector index | Semantic (vector) search | Narrative context, qualitative statements |

## How it's used

The [tools](../tools) wrap these access patterns so the [agent](../agents) can
call them:

- `tools/sql_tool.py` — structured retrieval over financials.
- `tools/document_search_tool.py` — semantic retrieval over document chunks.

A "hybrid" answer often uses both: pull the exact figures via SQL, then ground the
explanation in cited passages from the documents.

## Current implementation

Semantic retrieval is live in [`../tools/document_search_tool.py`](../tools/document_search_tool.py):

- **Similarity**: cosine distance over 512-dim model2vec embeddings, served by
  a pgvector **HNSW** index (`vector_cosine_ops`).
- **Filtering**: chunk metadata is JSONB with a GIN index; exact-match filters
  (`company_name`, `doc_type`, `section`, `content_kind`, `speakers`/`qa_role`)
  use containment (`@>`) inside the same query as the ANN search, plus
  `date_from`/`date_to` on `document_date` — no post-filtering.
- **Citations**: every result carries document id, title, section, date, and
  (for transcripts) speaker attribution.

> TODO: Evaluate re-ranking / hybrid (BM25 + vector) fusion once the agent
> layer generates real query traffic; static embeddings would benefit most.
