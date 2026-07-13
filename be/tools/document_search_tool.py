"""Semantic search over indexed document chunks, with citation metadata.

Vector search over the pgvector `document_chunks` table (cosine distance, HNSW
index) so the agent can ground answers in SEC filings and transcripts.

Filtered retrieval: exact-match filters map to JSONB containment on the chunk
metadata (backed by a GIN index), so the agent can scope a query precisely,
e.g. {"company_name": "Delta Air Lines", "doc_type": "10-K",
      "content_kind": "table"} — plus date_from/date_to on document_date.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Metadata keys the agent may filter on with exact-match containment.
FILTERABLE_KEYS = frozenset(
    {
        "company_name",
        "octus_company_id",
        "sub_industry",
        "source_type",
        "doc_type",
        "section",
        "content_kind",
        "qa_role",
    }
)


@dataclass
class SearchResult:
    """A single retrieved chunk plus the metadata needed to cite it.

    `document_id` is the source-system id; `title`/`metadata` carry the human
    citation context (company, doc type, date, section, speakers).
    """

    document_id: str
    chunk_index: int
    text: str
    score: float  # cosine similarity, higher = more relevant
    title: str | None
    metadata: dict[str, Any]


def document_coverage(database_url: str | None = None) -> dict[str, Any]:
    """What the document corpus actually covers — the doc-side discovery tool.

    The corpus is small (~12 companies) next to SimFin's 4,599, so an agent
    must scope claims like "sector trends" to what exists. This is the
    document analogue of the schema tools: real filter values instead of
    guesses.
    """
    from tools.sql_tool import run_read_only_sql

    per_company = run_read_only_sql(
        """
        SELECT d.company_name, c.ticker, d.sub_industry, d.doc_type,
               count(*) AS documents,
               min(d.document_date) AS first_date,
               max(d.document_date) AS last_date
        FROM documents d
        LEFT JOIN companies c ON c.id = d.company_id
        GROUP BY 1, 2, 3, 4
        ORDER BY d.company_name, d.doc_type
        """,
        database_url=database_url,
    )
    qa_roles = run_read_only_sql(
        """
        SELECT DISTINCT metadata->>'qa_role' AS qa_role
        FROM document_chunks
        WHERE metadata ? 'qa_role'
        ORDER BY 1
        """,
        database_url=database_url,
    )
    top_sections = run_read_only_sql(
        """
        SELECT metadata->>'section' AS section, count(*) AS chunks
        FROM document_chunks
        WHERE metadata ? 'section'
        GROUP BY 1
        ORDER BY count(*) DESC
        LIMIT 25
        """,
        database_url=database_url,
    )
    return {
        "coverage": list(per_company),
        "filterable_keys": sorted(FILTERABLE_KEYS) + ["date_from", "date_to"],
        "qa_roles": [r["qa_role"] for r in qa_roles],
        "top_sections": list(top_sections),
    }


def search_documents(
    query: str,
    *,
    k: int = 5,
    filters: dict[str, Any] | None = None,
    database_url: str | None = None,
) -> list[SearchResult]:
    """Find the `k` most relevant document chunks for a query.

    Args:
        query: Natural-language search query.
        k: Number of chunks to return.
        filters: Optional metadata filters. Keys in FILTERABLE_KEYS are
            exact-matched against chunk metadata; `date_from` / `date_to`
            (ISO dates) bound `document_date`.
        database_url: Override Postgres connection string.

    Returns:
        Ranked `SearchResult` objects, most relevant first.

    Raises:
        ValueError: on an unsupported filter key (typo protection — a silently
            ignored filter would return unfiltered results as if filtered).
    """
    from pgvector.psycopg import register_vector
    from psycopg.types.json import Jsonb

    from db.connection import get_connection
    from ingestion.embeddings import get_embedder

    filters = dict(filters or {})
    date_from = filters.pop("date_from", None)
    date_to = filters.pop("date_to", None)
    unknown = set(filters) - FILTERABLE_KEYS
    if unknown:
        raise ValueError(
            f"Unsupported filter key(s): {sorted(unknown)}. "
            f"Allowed: {sorted(FILTERABLE_KEYS)} plus date_from/date_to."
        )

    query_vector = get_embedder().embed([query])[0]

    where = ["c.embedding IS NOT NULL"]
    params: dict[str, Any] = {"q": query_vector, "k": k}
    if filters:
        where.append("c.metadata @> %(meta)s")
        params["meta"] = Jsonb(filters)
    if date_from:
        where.append("d.document_date >= %(date_from)s")
        params["date_from"] = date_from
    if date_to:
        where.append("d.document_date <= %(date_to)s")
        params["date_to"] = date_to

    sql = f"""
        SELECT d.external_id, d.title, c.chunk_index, c.content, c.metadata,
               1 - (c.embedding <=> %(q)s) AS score
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {" AND ".join(where)}
        ORDER BY c.embedding <=> %(q)s
        LIMIT %(k)s
    """

    with get_connection(database_url) as conn:
        conn.read_only = True  # retrieval must never write
        register_vector(conn)
        # Without this, HNSW fetches ~ef_search nearest candidates BEFORE the
        # metadata filter is applied — a selective filter (e.g. one company's
        # tables) can then match none of them and return 0 rows. Iterative
        # scanning (pgvector >= 0.8) keeps walking the graph until LIMIT is
        # satisfied, preserving exact ordering. Allowed in read-only txns.
        conn.execute("SET hnsw.iterative_scan = strict_order")
        rows = conn.execute(sql, params).fetchall()

    return [
        SearchResult(
            document_id=r["external_id"],
            chunk_index=r["chunk_index"],
            text=r["content"],
            score=float(r["score"]),
            title=r["title"],
            metadata=r["metadata"],
        )
        for r in rows
    ]
