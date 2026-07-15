"""Search over indexed document chunks, with citation metadata.

Three configurable retrieval strategies over the `document_chunks` table
(see `be/retrieval/README.md` for the design and `evals/retrieval.py` for the
benchmark that compares them):

- **dense**   — cosine distance over pgvector embeddings (HNSW index).
- **lexical** — PostgreSQL full-text search over `content_tsv` (GIN index),
  ranked with `ts_rank_cd`.
- **hybrid**  — both legs retrieved independently, fused with Reciprocal
  Rank Fusion (RRF). Raw cosine and ts_rank scores are never mixed directly —
  they are not on comparable scales; only ranks are combined.

Filtered retrieval: exact-match filters map to JSONB containment on the chunk
metadata (backed by a GIN index), so the agent can scope a query precisely,
e.g. {"company_name": "Delta Air Lines", "doc_type": "10-K",
      "content_kind": "table"} — plus date_from/date_to on document_date.
Every strategy applies the same filters to every leg.
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

RETRIEVAL_STRATEGIES = ("dense", "lexical", "hybrid")

# Cap on distinct lexemes used by the lexical quorum fallback: keeps the SQL
# bounded on pathological queries; realistic questions stopword-strip to far
# fewer. Truncation is deterministic (tsvector_to_array order).
_MAX_QUERY_LEXEMES = 12


@dataclass
class SearchResult:
    """A single retrieved chunk plus the metadata needed to cite it.

    `document_id` is the source-system id; `title`/`metadata` carry the human
    citation context (company, doc type, date, section, speakers).

    `score` semantics depend on `strategy` (higher = more relevant in all
    three): cosine similarity (dense), ts_rank_cd (lexical), or the fused RRF
    score (hybrid). The trailing trace fields exist for evaluation, debugging,
    and the frontend — they are deliberately NOT shown to the model
    (see agents/lc_tools.py, which projects only score + citation context).
    """

    document_id: str
    chunk_index: int
    text: str
    score: float
    title: str | None
    metadata: dict[str, Any]
    strategy: str = "dense"
    dense_rank: int | None = None
    lexical_rank: int | None = None
    fused_score: float | None = None


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


# ---------------------------------------------------------------------------
# Shared filter compilation
# ---------------------------------------------------------------------------

_RESULT_COLUMNS = "d.external_id, d.title, c.chunk_index, c.content, c.metadata"


def _compile_filters(
    filters: dict[str, Any] | None,
) -> tuple[list[str], dict[str, Any]]:
    """Turn the public filter dict into WHERE clauses + params.

    Shared by every retrieval leg so dense and lexical always apply identical
    metadata constraints.

    Raises:
        ValueError: on an unsupported filter key (typo protection — a silently
            ignored filter would return unfiltered results as if filtered).
    """
    from psycopg.types.json import Jsonb

    filters = dict(filters or {})
    date_from = filters.pop("date_from", None)
    date_to = filters.pop("date_to", None)
    unknown = set(filters) - FILTERABLE_KEYS
    if unknown:
        raise ValueError(
            f"Unsupported filter key(s): {sorted(unknown)}. "
            f"Allowed: {sorted(FILTERABLE_KEYS)} plus date_from/date_to."
        )

    where: list[str] = []
    params: dict[str, Any] = {}
    if filters:
        where.append("c.metadata @> %(meta)s")
        params["meta"] = Jsonb(filters)
    if date_from:
        where.append("d.document_date >= %(date_from)s")
        params["date_from"] = date_from
    if date_to:
        where.append("d.document_date <= %(date_to)s")
        params["date_to"] = date_to
    return where, params


def _rows_to_results(rows: list[dict[str, Any]], strategy: str) -> list[SearchResult]:
    return [
        SearchResult(
            document_id=r["external_id"],
            chunk_index=r["chunk_index"],
            text=r["content"],
            score=float(r["score"]),
            title=r["title"],
            metadata=r["metadata"],
            strategy=strategy,
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Retrieval legs
# ---------------------------------------------------------------------------


def _dense_leg(
    conn: Any, query: str, *, limit: int, where: list[str], params: dict[str, Any]
) -> list[SearchResult]:
    """pgvector cosine search — the pre-hybrid behaviour, unchanged."""
    from pgvector.psycopg import register_vector

    from ingestion.embeddings import get_embedder

    register_vector(conn)
    # Without this, HNSW fetches ~ef_search nearest candidates BEFORE the
    # metadata filter is applied — a selective filter (e.g. one company's
    # tables) can then match none of them and return 0 rows. Iterative
    # scanning (pgvector >= 0.8) keeps walking the graph until LIMIT is
    # satisfied, preserving exact ordering. Allowed in read-only txns.
    conn.execute("SET hnsw.iterative_scan = strict_order")

    sql = f"""
        SELECT {_RESULT_COLUMNS},
               1 - (c.embedding <=> %(q)s) AS score
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {" AND ".join(["c.embedding IS NOT NULL", *where])}
        ORDER BY c.embedding <=> %(q)s
        LIMIT %(k)s
    """
    query_vector = get_embedder().embed([query])[0]
    rows = conn.execute(sql, {**params, "q": query_vector, "k": limit}).fetchall()
    results = _rows_to_results(rows, "dense")
    for rank, r in enumerate(results, start=1):
        r.dense_rank = rank
    return results


def _lexical_leg(
    conn: Any, query: str, *, limit: int, where: list[str], params: dict[str, Any]
) -> list[SearchResult]:
    """Postgres full-text search, ranked by ts_rank_cd.

    Query semantics: `websearch_to_tsquery` first — it is safe on arbitrary
    user text and preserves phrases, so hyphenated jargon ("CASM-ex") matches
    exactly. But it ANDs every non-stopword, and a long natural-language
    query ("what did management say about fuel hedging") rarely has ALL its
    terms in one chunk — strict AND would return nothing. When that happens,
    fall back to OR-ing the query's lexemes (disjunctive matching).

    The fallback ranks by **quorum first** (how many distinct query lexemes a
    chunk matches), then ts_rank_cd. Plain ts_rank_cd is term-frequency-only —
    Postgres FTS has no IDF — so for "How did Delta's TRASM trend?" a chunk
    repeating the common term ("Delta" 30×) would outrank every chunk
    containing the rare, high-signal term (TRASM). The retrieval benchmark
    caught exactly that failure (case ex-trasm-delta); quorum ordering fixes
    it with stock operators, not custom tokenization. Deterministic: same
    query text always takes the same path, and all ties break by
    (external_id, chunk_index).
    """
    forms = conn.execute(
        """
        SELECT websearch_to_tsquery('english', %(q)s)::text AS strict,
               tsvector_to_array(to_tsvector('english', %(q)s)) AS lexemes
        """,
        {"q": query},
    ).fetchone()

    # NOTE: ordering is (matched_terms, score); SearchResult.score reports the
    # ts_rank_cd component, so in the quorum fallback a result with a lower
    # score can legitimately rank above a higher one that matched fewer terms.
    base_sql = """
        SELECT {columns},
               {quorum} AS matched_terms,
               ts_rank_cd(c.content_tsv, tsq) AS score
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id,
             to_tsquery('english', %(tsq)s) tsq
        WHERE {conditions}
        ORDER BY matched_terms DESC, score DESC,
                 d.external_id, c.chunk_index
        LIMIT %(k)s
    """
    conditions = " AND ".join(["c.content_tsv @@ tsq", *where])

    rows: list[dict[str, Any]] = []
    if forms["strict"]:
        # Strict pass: every result matches all terms — quorum is constant.
        sql = base_sql.format(
            columns=_RESULT_COLUMNS, quorum="1", conditions=conditions
        )
        rows = conn.execute(
            sql, {**params, "tsq": forms["strict"], "k": limit}
        ).fetchall()

    lexemes = [lex for lex in (forms["lexemes"] or []) if lex][:_MAX_QUERY_LEXEMES]
    if not rows and len(lexemes) > 1:
        # to_tsquery treats a single-quoted token as a literal lexeme, which
        # keeps compounds like 'casm-ex' intact through the round trip.
        quoted = ["'" + lex.replace("'", "''") + "'" for lex in lexemes]
        quorum = " + ".join(
            f"(c.content_tsv @@ to_tsquery('english', %(lex_{i})s))::int"
            for i in range(len(quoted))
        )
        lex_params = {f"lex_{i}": q for i, q in enumerate(quoted)}
        sql = base_sql.format(
            columns=_RESULT_COLUMNS, quorum=f"({quorum})", conditions=conditions
        )
        rows = conn.execute(
            sql,
            {**params, **lex_params, "tsq": " | ".join(quoted), "k": limit},
        ).fetchall()
    results = _rows_to_results(rows, "lexical")
    for rank, r in enumerate(results, start=1):
        r.lexical_rank = rank
    return results


def _rrf_fuse(
    dense: list[SearchResult],
    lexical: list[SearchResult],
    *,
    k: int,
    rrf_k: int,
) -> list[SearchResult]:
    """Fuse two ranked lists with Reciprocal Rank Fusion.

    score(chunk) = Σ_legs 1 / (rrf_k + rank_in_leg); a chunk missing from a
    leg contributes nothing for that leg. Results are deduplicated by
    (document_id, chunk_index) and ordered by fused score with a
    deterministic (document_id, chunk_index) tie-break.
    """
    fused: dict[tuple[str, int], SearchResult] = {}
    for leg_name, leg in (("dense", dense), ("lexical", lexical)):
        for rank, r in enumerate(leg, start=1):
            key = (r.document_id, r.chunk_index)
            entry = fused.get(key)
            if entry is None:
                # Copy so leg results stay untouched; text/metadata are
                # identical for the same chunk whichever leg found it first.
                entry = fused[key] = SearchResult(
                    document_id=r.document_id,
                    chunk_index=r.chunk_index,
                    text=r.text,
                    score=0.0,
                    title=r.title,
                    metadata=r.metadata,
                    strategy="hybrid",
                    fused_score=0.0,
                )
            if leg_name == "dense":
                entry.dense_rank = rank
            else:
                entry.lexical_rank = rank
            entry.fused_score += 1.0 / (rrf_k + rank)
            entry.score = entry.fused_score

    ordered = sorted(
        fused.values(),
        key=lambda r: (-r.fused_score, r.document_id, r.chunk_index),
    )
    return ordered[:k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_documents(
    query: str,
    *,
    k: int = 5,
    filters: dict[str, Any] | None = None,
    database_url: str | None = None,
    strategy: str | None = None,
    leg_k: int | None = None,
    rrf_k: int | None = None,
) -> list[SearchResult]:
    """Find the `k` most relevant document chunks for a query.

    Args:
        query: Natural-language search query.
        k: Number of chunks to return.
        filters: Optional metadata filters. Keys in FILTERABLE_KEYS are
            exact-matched against chunk metadata; `date_from` / `date_to`
            (ISO dates) bound `document_date`. Applied identically to every
            retrieval leg.
        database_url: Override Postgres connection string.
        strategy: 'dense' | 'lexical' | 'hybrid'; defaults to
            `settings.retrieval_strategy`.
        leg_k: Per-leg candidate depth for hybrid fusion; defaults to
            `settings.retrieval_leg_k`. Ignored by dense/lexical.
        rrf_k: RRF constant; defaults to `settings.retrieval_rrf_k`.
            Ignored by dense/lexical.

    Returns:
        Ranked `SearchResult` objects, most relevant first, in deterministic
        order. Each carries its retrieval trace (strategy, per-leg ranks,
        fused score) for evaluation/debugging — see SearchResult.

    Raises:
        ValueError: on an unsupported filter key, an unknown strategy, or a
            non-positive k / leg_k / rrf_k.
    """
    from core.config import settings
    from db.connection import get_connection

    strategy = strategy or settings.retrieval_strategy
    if strategy not in RETRIEVAL_STRATEGIES:
        raise ValueError(
            f"Unknown retrieval strategy {strategy!r}. "
            f"Allowed: {list(RETRIEVAL_STRATEGIES)}."
        )
    leg_k = leg_k if leg_k is not None else settings.retrieval_leg_k
    rrf_k = rrf_k if rrf_k is not None else settings.retrieval_rrf_k
    if k < 1 or leg_k < 1 or rrf_k < 1:
        raise ValueError(
            f"k ({k}), leg_k ({leg_k}) and rrf_k ({rrf_k}) must all be >= 1."
        )

    where, params = _compile_filters(filters)

    with get_connection(database_url) as conn:
        conn.read_only = True  # retrieval must never write
        if strategy == "dense":
            return _dense_leg(conn, query, limit=k, where=where, params=params)
        if strategy == "lexical":
            return _lexical_leg(conn, query, limit=k, where=where, params=params)
        dense = _dense_leg(conn, query, limit=leg_k, where=where, params=params)
        lexical = _lexical_leg(conn, query, limit=leg_k, where=where, params=params)
    return _rrf_fuse(dense, lexical, k=k, rrf_k=rrf_k)
