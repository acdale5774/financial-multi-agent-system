"""Tests for the dense/lexical/hybrid retrieval strategies and their benchmark.

Three tiers, following the suite's conventions (see tests/README.md):

- **Pure unit** — RRF fusion math, deduplication, tie-breaking, configuration
  validation, benchmark-case well-formedness, and metric calculations. These
  prove logic only.
- **Mocked dispatch** — strategy selection drives the right legs, with the
  legs and the connection monkeypatched. Proves wiring, NOT retrieval quality.
- **Live integration** — real Postgres full-text + vector queries; auto-skips
  when Postgres isn't reachable. Only this tier says anything about actual
  retrieval behaviour; the benchmark (`python -m evals.retrieval`) is what
  measures retrieval *quality*.
"""

from __future__ import annotations

from typing import Any

import pytest

from tools.document_search_tool import (
    FILTERABLE_KEYS,
    SearchResult,
    _compile_filters,
    _rrf_fuse,
    search_documents,
)


def _result(doc: str, chunk: int, **kwargs: Any) -> SearchResult:
    defaults: dict[str, Any] = dict(
        text=f"chunk {doc}/{chunk}",
        score=1.0,
        title=None,
        metadata={"company_name": "TestCo", "section": "Item 1A"},
    )
    defaults.update(kwargs)
    return SearchResult(document_id=doc, chunk_index=chunk, **defaults)


def _database_available() -> bool:
    try:
        from tools.sql_tool import run_read_only_sql

        run_read_only_sql("SELECT 1 AS answer", statement_timeout_ms=3000)
        return True
    except Exception:  # noqa: BLE001 - any failure means "no live DB for this test"
        return False


# ---------------------------------------------------------------------------
# RRF fusion (pure)
# ---------------------------------------------------------------------------


def test_rrf_scores_sum_across_legs() -> None:
    dense = [_result("a", 1), _result("b", 1)]
    lexical = [_result("b", 1), _result("c", 1)]
    fused = _rrf_fuse(dense, lexical, k=10, rrf_k=60)

    by_key = {(r.document_id, r.chunk_index): r for r in fused}
    # 'b' appears in both legs: dense rank 2, lexical rank 1.
    b = by_key[("b", 1)]
    assert b.fused_score == pytest.approx(1 / 62 + 1 / 61)
    assert (b.dense_rank, b.lexical_rank) == (2, 1)
    # 'a' is dense-only (rank 1): the missing lexical leg contributes nothing.
    a = by_key[("a", 1)]
    assert a.fused_score == pytest.approx(1 / 61)
    assert (a.dense_rank, a.lexical_rank) == (1, None)
    # Two-leg consensus beats either single leg.
    assert fused[0] is b


def test_rrf_deduplicates_by_chunk_identity() -> None:
    dense = [_result("a", 7)]
    lexical = [_result("a", 7)]
    fused = _rrf_fuse(dense, lexical, k=10, rrf_k=60)
    assert len(fused) == 1
    assert fused[0].fused_score == pytest.approx(2 / 61)


def test_rrf_ties_break_deterministically_by_id() -> None:
    # Symmetric ranks across legs -> identical fused scores for both chunks.
    dense = [_result("b", 2), _result("a", 9)]
    lexical = [_result("a", 9), _result("b", 2)]
    fused = _rrf_fuse(dense, lexical, k=10, rrf_k=60)
    # Both chunks: 1/(60+1) + 1/(60+2) — identical fused scores.
    assert fused[0].fused_score == pytest.approx(fused[1].fused_score)
    # Tie broken by (document_id, chunk_index): 'a' before 'b'.
    assert [(r.document_id, r.chunk_index) for r in fused] == [("a", 9), ("b", 2)]


def test_rrf_truncates_to_k_and_marks_strategy() -> None:
    dense = [_result(f"d{i}", i) for i in range(1, 8)]
    fused = _rrf_fuse(dense, [], k=3, rrf_k=60)
    assert len(fused) == 3
    assert all(r.strategy == "hybrid" for r in fused)
    assert all(r.score == r.fused_score for r in fused)
    # Empty lexical leg: pure dense ordering is preserved.
    assert [r.document_id for r in fused] == ["d1", "d2", "d3"]


def test_rrf_preserves_citation_metadata() -> None:
    """The evidence-ledger contract: fused results keep the fields
    record_documents() needs to mint citations."""
    dense = [
        _result(
            "doc-1",
            4,
            title="TestCo 10-K",
            metadata={
                "company_name": "TestCo",
                "doc_type": "10-K",
                "section": "Item 1A",
                "document_date": "2024-01-01",
                "speakers": ["A. Person"],
            },
        )
    ]
    fused = _rrf_fuse(dense, [], k=5, rrf_k=60)
    r = fused[0]
    assert r.title == "TestCo 10-K"
    assert r.metadata["doc_type"] == "10-K"
    assert r.metadata["speakers"] == ["A. Person"]
    assert r.text == "chunk doc-1/4"


def test_ledger_mints_citations_from_hybrid_results() -> None:
    from agents.ledger import EvidenceLedger

    ledger = EvidenceLedger()
    results = _rrf_fuse([_result("doc-9", 2)], [_result("doc-9", 2)], k=5, rrf_k=60)
    minted = ledger.record_documents("q", None, results)
    assert len(minted) == 1
    assert minted[0].id.startswith("D")
    assert minted[0].document_id == "doc-9"
    assert minted[0].chunk_index == 2


# ---------------------------------------------------------------------------
# Configuration validation (pure — errors raise before any DB access)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["bm25", "sparse", "DENSE"])
def test_unknown_strategy_rejected(strategy: str) -> None:
    # (strategy="" falls back to the configured default by design — the
    # falsy sentinel is how callers say "use settings.retrieval_strategy".)
    with pytest.raises(ValueError, match="retrieval strategy"):
        search_documents("q", strategy=strategy)


@pytest.mark.parametrize(
    "kwargs", [{"k": 0}, {"leg_k": 0}, {"rrf_k": 0}, {"k": -3}]
)
def test_non_positive_knobs_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must all be >= 1"):
        search_documents("q", strategy="hybrid", **kwargs)


def test_unsupported_filter_key_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported filter key"):
        _compile_filters({"companyname": "typo"})


def test_filter_compilation_shared_by_all_legs() -> None:
    where, params = _compile_filters(
        {"company_name": "TestCo", "date_from": "2024-01-01"}
    )
    assert any("metadata @>" in w for w in where)
    assert any("document_date >=" in w for w in where)
    assert params["date_from"] == "2024-01-01"


def test_search_result_backward_compatible_construction() -> None:
    # Pre-hybrid callers construct with six positional fields.
    r = SearchResult("doc", 1, "text", 0.5, None, {})
    assert r.strategy == "dense"
    assert r.dense_rank is None and r.lexical_rank is None
    assert r.fused_score is None


# ---------------------------------------------------------------------------
# Strategy dispatch (mocked legs; proves wiring, not quality)
# ---------------------------------------------------------------------------


class _FakeConn:
    read_only = False

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


@pytest.fixture
def dispatch_probe(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"dense": 0, "lexical": 0, "limits": {}}

    def fake_dense(conn: Any, query: str, *, limit: int, where: Any, params: Any):
        calls["dense"] += 1
        calls["limits"]["dense"] = limit
        return [_result("d", 1, strategy="dense", dense_rank=1)]

    def fake_lexical(conn: Any, query: str, *, limit: int, where: Any, params: Any):
        calls["lexical"] += 1
        calls["limits"]["lexical"] = limit
        return [_result("l", 1, strategy="lexical", lexical_rank=1)]

    import db.connection
    import tools.document_search_tool as dst

    monkeypatch.setattr(dst, "_dense_leg", fake_dense)
    monkeypatch.setattr(dst, "_lexical_leg", fake_lexical)
    monkeypatch.setattr(db.connection, "get_connection", lambda *a, **k: _FakeConn())
    return calls


def test_dense_strategy_runs_only_dense_leg(dispatch_probe: dict[str, Any]) -> None:
    results = search_documents("q", k=5, strategy="dense")
    assert dispatch_probe == {"dense": 1, "lexical": 0, "limits": {"dense": 5}}
    assert [r.document_id for r in results] == ["d"]


def test_lexical_strategy_runs_only_lexical_leg(dispatch_probe: dict[str, Any]) -> None:
    results = search_documents("q", k=5, strategy="lexical")
    assert dispatch_probe["dense"] == 0 and dispatch_probe["lexical"] == 1
    assert [r.document_id for r in results] == ["l"]


def test_hybrid_strategy_fuses_both_legs_at_leg_k(
    dispatch_probe: dict[str, Any],
) -> None:
    results = search_documents("q", k=2, strategy="hybrid", leg_k=25, rrf_k=60)
    assert dispatch_probe["dense"] == 1 and dispatch_probe["lexical"] == 1
    # Both legs are fetched at leg_k depth, not final k.
    assert dispatch_probe["limits"] == {"dense": 25, "lexical": 25}
    assert {r.document_id for r in results} == {"d", "l"}
    assert all(r.strategy == "hybrid" for r in results)


# ---------------------------------------------------------------------------
# Benchmark case well-formedness (pure)
# ---------------------------------------------------------------------------


def test_retrieval_cases_are_well_formed() -> None:
    from evals.retrieval_cases import CATEGORIES, RETRIEVAL_CASES

    ids = [c.id for c in RETRIEVAL_CASES]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    assert 20 <= len(RETRIEVAL_CASES) <= 40, "benchmark should stay small/inspectable"

    allowed_filters = FILTERABLE_KEYS | {"date_from", "date_to"}
    for case in RETRIEVAL_CASES:
        assert case.category in CATEGORIES, case.id
        assert case.query.strip(), case.id
        assert case.why.strip(), case.id
        assert set(case.filters) <= allowed_filters, case.id
        if case.category == "unsupported":
            assert not case.anchors, f"{case.id}: unsupported ⇔ no anchors"
        else:
            assert case.anchors, f"{case.id}: supported cases need anchors"


def test_retrieval_cases_cover_every_category() -> None:
    from evals.retrieval_cases import CATEGORIES, RETRIEVAL_CASES

    covered = {c.category for c in RETRIEVAL_CASES}
    assert covered == set(CATEGORIES)


# ---------------------------------------------------------------------------
# Metric calculations (pure)
# ---------------------------------------------------------------------------


def _case_result(category: str, rank: int | None, results: int = 10):
    from evals.retrieval import CaseResult
    from evals.retrieval_cases import RetrievalCase

    case = RetrievalCase(
        id=f"c-{category}-{rank}",
        category=category,
        query="q",
        why="w",
        anchors=() if category == "unsupported" else ("x",),
    )
    return CaseResult(
        case=case,
        strategy="dense",
        relevant_count=0 if category == "unsupported" else 3,
        result_count=results,
        first_relevant_rank=rank,
        relevant_ranks=[rank] if rank else [],
    )


def test_metric_calculations() -> None:
    from evals.retrieval import summarize

    results = [
        _case_result("semantic", 1),       # hit@5, rr=1
        _case_result("semantic", 7),       # miss@5, hit@10, rr=1/7
        _case_result("exact_term", None),  # miss, rr=0
        _case_result("exact_term", None, results=0),  # zero results
        _case_result("unsupported", None, results=10),
        _case_result("unsupported", None, results=0),
    ]
    m = summarize(results)
    assert m["supported_cases"] == 4
    assert m["recall@5"] == pytest.approx(1 / 4)
    assert m["recall@10"] == pytest.approx(2 / 4)
    assert m["mrr"] == pytest.approx((1 + 1 / 7) / 4)
    assert m["zero_result_rate"] == pytest.approx(1 / 4)
    assert m["by_category"]["semantic"]["recall@10"] == pytest.approx(1.0)
    assert m["by_category"]["exact_term"]["mrr"] == 0.0
    assert m["unsupported"]["cases"] == 2
    assert m["unsupported"]["mean_results_returned"] == pytest.approx(5.0)
    assert m["unsupported"]["zero_result_cases"] == 1


def test_evaluate_case_grades_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    from evals.retrieval import evaluate_case
    from evals.retrieval_cases import RetrievalCase

    case = RetrievalCase(
        id="t", category="semantic", query="q", why="w", anchors=("x",)
    )
    canned = [_result("keep", 1), _result("skip", 2), _result("keep", 3)]
    # evaluate_case imports search_documents lazily, so patch its source module.
    monkeypatch.setattr(
        "tools.document_search_tool.search_documents", lambda *a, **k: canned
    )
    r = evaluate_case(case, "dense", {("keep", 1), ("keep", 3)})
    assert r.first_relevant_rank == 1
    assert r.relevant_ranks == [1, 3]
    assert r.result_count == 3
    assert r.hit(5) and r.reciprocal_rank == 1.0


# ---------------------------------------------------------------------------
# Live integration (auto-skips without Postgres; same pattern as test_evals)
# ---------------------------------------------------------------------------

requires_db = pytest.mark.skipif(
    not _database_available(), reason="Postgres not reachable"
)


def _query_embedder_available() -> bool:
    """The dense leg embeds the query text, so with EMBEDDING_PROVIDER=openai
    the dense/hybrid live tests also need an API key (a few embedding calls,
    fractions of a cent). Lexical live tests stay Postgres-only."""
    import os

    from core.config import settings

    if settings.embedding_provider != "openai":
        return True
    return bool(settings.openai_api_key or os.environ.get("OPENAI_API_KEY"))


requires_query_embedder = pytest.mark.skipif(
    not _query_embedder_available(),
    reason="EMBEDDING_PROVIDER=openai needs OPENAI_API_KEY to embed queries",
)


@requires_db
def test_live_lexical_finds_exact_term() -> None:
    results = search_documents("TRASM", k=5, strategy="lexical")
    assert results, "corpus contains TRASM chunks; lexical must find them"
    assert all("TRASM" in r.text.upper() for r in results)
    assert [r.lexical_rank for r in results] == list(range(1, len(results) + 1))


@requires_db
def test_live_lexical_applies_metadata_filters() -> None:
    results = search_documents(
        "revenue",
        k=5,
        strategy="lexical",
        filters={"company_name": "Delta Air Lines", "doc_type": "10-K"},
    )
    assert results
    for r in results:
        assert r.metadata["company_name"] == "Delta Air Lines"
        assert r.metadata["doc_type"] == "10-K"


@requires_db
@requires_query_embedder
def test_live_dense_behaviour_unchanged() -> None:
    """Dense mode is the pre-hybrid code path: cosine scores, descending."""
    results = search_documents("fuel cost hedging", k=5, strategy="dense")
    assert len(results) == 5
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores), "cosine similarity range"
    assert all(r.lexical_rank is None for r in results)
    assert all(r.strategy == "dense" for r in results)


@requires_db
@requires_query_embedder
def test_live_hybrid_is_deduplicated_and_deterministic() -> None:
    kwargs: dict[str, Any] = dict(k=10, strategy="hybrid")
    first = search_documents("JetBlue CASM ex-fuel outlook", **kwargs)
    second = search_documents("JetBlue CASM ex-fuel outlook", **kwargs)

    keys = [(r.document_id, r.chunk_index) for r in first]
    assert len(keys) == len(set(keys)), "no duplicate chunks after fusion"
    assert keys == [(r.document_id, r.chunk_index) for r in second]
    fused = [r.fused_score for r in first]
    assert fused == sorted(fused, reverse=True)


@requires_db
def test_live_lexical_all_stopword_query_returns_empty() -> None:
    # Every term is an english-config stopword -> no tsquery can be built.
    assert search_documents("the of and", k=5, strategy="lexical") == []
