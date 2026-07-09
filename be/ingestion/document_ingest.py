"""Ingest unstructured documents (SEC filings, earnings transcripts) into pgvector.

Pipeline:
    chunk_document()  ->  embed_and_index_chunks()

Chunks retain enough metadata (source document, section, offsets) to produce
citations when the agent answers a question.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """A single retrievable slice of a source document.

    Attributes:
        document_id: Identifier of the parent document.
        chunk_index: Position of this chunk within the document.
        text: The chunk's plain text content.
        metadata: Free-form context used for filtering and citations
            (e.g. company, filing type, fiscal period, section, char offsets).
    """

    document_id: str
    chunk_index: int
    text: str
    metadata: dict[str, str]


def chunk_document(
    text: str,
    *,
    document_id: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[DocumentChunk]:
    """Split a document into overlapping, embedding-sized chunks.

    Args:
        text: Full plain-text content of the document.
        document_id: Identifier of the parent document (for citations).
        max_tokens: Target maximum chunk size in tokens.
        overlap_tokens: Token overlap between adjacent chunks to preserve context.

    Returns:
        Ordered list of `DocumentChunk` objects.

    TODO: Use a real tokenizer and a structure-aware splitter (respect section
          and paragraph boundaries) instead of naive fixed-size windows.
    """
    raise NotImplementedError


def embed_and_index_chunks(
    chunks: list[DocumentChunk],
    *,
    database_url: str | None = None,
) -> int:
    """Embed chunks and store them in the pgvector `document_chunks` table.

    Args:
        chunks: Chunks produced by `chunk_document`.
        database_url: Postgres connection string; falls back to DATABASE_URL.

    Returns:
        The number of chunks embedded and indexed.

    TODO: Choose an embedding provider/model and record its dimensionality
          (must match the vector column in db/schema.sql). Batch embedding
          calls, then bulk-insert text + metadata + embedding in one transaction.
    """
    raise NotImplementedError
