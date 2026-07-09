"""Semantic search over indexed document chunks, with citation metadata.

Wraps vector search over the pgvector `document_chunks` table so the agent can
ground answers in SEC filings and earnings-call transcripts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchResult:
    """A single retrieved chunk plus the metadata needed to cite it.

    Attributes:
        document_id: Identifier of the source document.
        chunk_index: Position of the chunk within its document.
        text: The retrieved chunk text.
        score: Similarity score (higher = more relevant).
        metadata: Citation context (e.g. company, filing type, fiscal period,
            section, char offsets, source URL).
    """

    document_id: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, str]


def search_documents(
    query: str,
    *,
    k: int = 5,
    filters: dict[str, str] | None = None,
    database_url: str | None = None,
) -> list[SearchResult]:
    """Find the `k` most relevant document chunks for a query.

    Args:
        query: Natural-language search query.
        k: Number of chunks to return.
        filters: Optional metadata filters (e.g. company, filing type, period).
        database_url: Postgres connection string; falls back to DATABASE_URL.

    Returns:
        Ranked `SearchResult` objects, most relevant first.

    TODO: Embed `query` with the same model used at ingest time, then run a
          nearest-neighbor search against the pgvector index (ORDER BY embedding
          <-> :q LIMIT k) with optional metadata filtering.
    """
    raise NotImplementedError
