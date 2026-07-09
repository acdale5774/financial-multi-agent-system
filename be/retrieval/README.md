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

## Planned structure (not implemented yet)

- `vector_search.py` — nearest-neighbor search + metadata filtering over pgvector.
- `ranking.py` — optional re-ranking / fusion of structured + semantic results.

> TODO: Decide whether re-ranking is needed and document the retrieval strategy
> (top-k, similarity metric, metadata filters) here.
