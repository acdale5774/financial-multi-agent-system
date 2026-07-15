"""Runner + scorecard for the labeled document-retrieval benchmark.

Compares the dense, lexical, and hybrid strategies of
`tools.document_search_tool.search_documents` against the SAME labeled cases
(`evals/retrieval_cases.py`), so differences are attributable to the strategy,
not the data.

Matching rule (explicit and deterministic):
    Ground truth per case = the set of (document external_id, chunk_index)
    pairs whose chunk satisfies the case's document constraints AND contains
    at least one anchor phrase (case-insensitive substring), resolved with
    plain SQL — the retriever is never consulted to build labels.
    A retrieved chunk counts as relevant iff its (external_id, chunk_index)
    is in that set.

Metrics (per strategy, over the supported cases):
    Recall@k    fraction of cases with >= 1 relevant chunk in the top k
                (hit rate — each case's evidence set is defined so that any
                one of its chunks answers the query).
    MRR         mean of 1/rank of the first relevant chunk in the top K
                (0 when none appears).
    zero-result rate   fraction of cases returning no results at all.

`unsupported` cases (no relevant evidence exists in the corpus) are excluded
from recall/MRR and reported separately: the honest behaviour is to return
nothing that pretends to be evidence, and the strategies differ interestingly
here — vector search always returns *something*, full-text search can
return nothing.

Run it (needs Postgres; no LLM. With EMBEDDING_PROVIDER=openai the dense and
hybrid legs embed each case's query via the API — 29 embedding calls,
fractions of a cent; `--strategies lexical` stays key-free):
    python -m evals.retrieval                      # all three strategies
    python -m evals.retrieval --strategies hybrid  # one strategy
    python -m evals.retrieval --write-report       # refresh the committed
                                                   # retrieval_report.{md,json}

Exit code is non-zero if any supported case's ground-truth set resolves
empty — that means the labels drifted from the corpus (same convention as
the text-to-SQL harness treating golden-SQL drift as a build break).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals.retrieval_cases import CATEGORIES, RETRIEVAL_CASES, RetrievalCase

# Retrieval depth: metrics are computed at cutoffs <= this.
K_MAX = 10
RECALL_CUTOFFS = (5, 10)

_REPORT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Ground truth resolution (SQL only — never the retriever)
# ---------------------------------------------------------------------------


def resolve_relevant_chunks(
    case: RetrievalCase, *, database_url: str | None = None
) -> set[tuple[str, int]]:
    """Resolve a case's declarative relevance definition against the corpus."""
    if not case.anchors:
        return set()

    from db.connection import get_connection

    where = ["(" + " OR ".join(
        f"c.content ILIKE %(anchor_{i})s" for i in range(len(case.anchors))
    ) + ")"]
    params: dict[str, Any] = {
        f"anchor_{i}": f"%{a}%" for i, a in enumerate(case.anchors)
    }
    if case.relevant_companies:
        where.append("d.company_name = ANY(%(companies)s)")
        params["companies"] = list(case.relevant_companies)
    if case.relevant_doc_type:
        where.append("d.doc_type = %(doc_type)s")
        params["doc_type"] = case.relevant_doc_type
    if case.relevant_section_ilike:
        where.append("c.metadata->>'section' ILIKE %(section)s")
        params["section"] = case.relevant_section_ilike
    if case.relevant_date_from:
        where.append("d.document_date >= %(date_from)s")
        params["date_from"] = case.relevant_date_from
    if case.relevant_date_to:
        where.append("d.document_date <= %(date_to)s")
        params["date_to"] = case.relevant_date_to
    if case.relevant_content_kind:
        where.append("c.metadata->>'content_kind' = %(content_kind)s")
        params["content_kind"] = case.relevant_content_kind
    if case.relevant_speaker:
        where.append("c.metadata->>'speakers' ILIKE %(speaker)s")
        params["speaker"] = f"%{case.relevant_speaker}%"
    if case.relevant_qa_role:
        where.append("c.metadata->>'qa_role' = %(qa_role)s")
        params["qa_role"] = case.relevant_qa_role

    sql = f"""
        SELECT d.external_id, c.chunk_index
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {" AND ".join(where)}
    """
    with get_connection(database_url) as conn:
        conn.read_only = True
        rows = conn.execute(sql, params).fetchall()
    return {(r["external_id"], r["chunk_index"]) for r in rows}


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    """One (case, strategy) evaluation outcome."""

    case: RetrievalCase
    strategy: str
    relevant_count: int  # size of the ground-truth set in the corpus
    result_count: int  # how many chunks the strategy returned
    first_relevant_rank: int | None  # 1-based; None = no relevant in top K_MAX
    relevant_ranks: list[int] = field(default_factory=list)
    error: str | None = None

    def hit(self, k: int) -> bool:
        return self.first_relevant_rank is not None and self.first_relevant_rank <= k

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_relevant_rank if self.first_relevant_rank else 0.0


def evaluate_case(
    case: RetrievalCase,
    strategy: str,
    relevant: set[tuple[str, int]],
    *,
    database_url: str | None = None,
) -> CaseResult:
    """Run one strategy on one case and grade the ranked results."""
    from tools.document_search_tool import search_documents

    try:
        results = search_documents(
            case.query,
            k=K_MAX,
            filters=case.filters or None,
            strategy=strategy,
            database_url=database_url,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced per-case, never aborts the sweep
        return CaseResult(
            case=case,
            strategy=strategy,
            relevant_count=len(relevant),
            result_count=0,
            first_relevant_rank=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    ranks = [
        rank
        for rank, r in enumerate(results, start=1)
        if (r.document_id, r.chunk_index) in relevant
    ]
    return CaseResult(
        case=case,
        strategy=strategy,
        relevant_count=len(relevant),
        result_count=len(results),
        first_relevant_rank=ranks[0] if ranks else None,
        relevant_ranks=ranks,
    )


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def run_benchmark(
    strategies: list[str],
    *,
    cases: list[RetrievalCase] | None = None,
    database_url: str | None = None,
) -> tuple[dict[str, list[CaseResult]], list[str]]:
    """Evaluate every strategy against every case.

    Returns:
        ({strategy: [CaseResult, ...]}, label_errors) — `label_errors` lists
        supported cases whose ground-truth set resolved empty (label drift;
        those cases are skipped and the run should exit non-zero).
    """
    cases = cases if cases is not None else RETRIEVAL_CASES
    relevant_sets: dict[str, set[tuple[str, int]]] = {}
    label_errors: list[str] = []
    for case in cases:
        relevant_sets[case.id] = resolve_relevant_chunks(
            case, database_url=database_url
        )
        if case.category != "unsupported" and not relevant_sets[case.id]:
            label_errors.append(
                f"{case.id}: ground-truth set resolved empty — anchors "
                f"{case.anchors!r} no longer match the corpus."
            )

    graded_cases = [
        c
        for c in cases
        if c.category == "unsupported" or relevant_sets[c.id]
    ]
    by_strategy: dict[str, list[CaseResult]] = {}
    for strategy in strategies:
        by_strategy[strategy] = [
            evaluate_case(
                case, strategy, relevant_sets[case.id], database_url=database_url
            )
            for case in graded_cases
        ]
    return by_strategy, label_errors


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    """Aggregate one strategy's results into the scorecard metrics."""
    supported = [r for r in results if r.case.category != "unsupported"]
    unsupported = [r for r in results if r.case.category == "unsupported"]

    def recall_at(rs: list[CaseResult], k: int) -> float:
        return sum(r.hit(k) for r in rs) / len(rs) if rs else 0.0

    def mrr(rs: list[CaseResult]) -> float:
        return sum(r.reciprocal_rank for r in rs) / len(rs) if rs else 0.0

    by_category: dict[str, dict[str, float]] = {}
    for category in CATEGORIES:
        if category == "unsupported":
            continue
        bucket = [r for r in supported if r.case.category == category]
        if bucket:
            by_category[category] = {
                "cases": len(bucket),
                **{f"recall@{k}": recall_at(bucket, k) for k in RECALL_CUTOFFS},
                "mrr": mrr(bucket),
            }

    return {
        "supported_cases": len(supported),
        **{f"recall@{k}": recall_at(supported, k) for k in RECALL_CUTOFFS},
        "mrr": mrr(supported),
        "zero_result_rate": (
            sum(r.result_count == 0 for r in supported) / len(supported)
            if supported
            else 0.0
        ),
        "errors": [r.case.id for r in results if r.error],
        "by_category": by_category,
        "unsupported": {
            "cases": len(unsupported),
            "mean_results_returned": (
                sum(r.result_count for r in unsupported) / len(unsupported)
                if unsupported
                else 0.0
            ),
            "zero_result_cases": sum(r.result_count == 0 for r in unsupported),
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def format_markdown(
    by_strategy: dict[str, list[CaseResult]],
    summaries: dict[str, dict[str, Any]],
    *,
    run_date: str,
    corpus_note: str,
    config_note: str,
) -> str:
    """Render the human-readable scorecard."""
    strategies = list(by_strategy)
    lines: list[str] = []
    add = lines.append

    add("# Document-retrieval benchmark report")
    add("")
    add(f"Run: {run_date} · {corpus_note} · {config_note}")
    add("")
    add(
        "Generated by `python -m evals.retrieval --write-report`. "
        "Labels and methodology: `evals/retrieval_cases.py`. "
        "**Read the limitations there before quoting these numbers** — "
        "29 cases compares strategies directionally; it does not certify "
        "production recall. The default-strategy selection rationale lives "
        "in `be/retrieval/README.md` (kept out of this generated file so "
        "regeneration can't silently rewrite a decision)."
    )
    add("")
    add("## Matching rule")
    add("")
    add(
        "A retrieved chunk is relevant iff its (document external_id, "
        "chunk_index) is in the case's ground-truth set — the chunks "
        "satisfying the case's document constraints AND containing an anchor "
        "phrase, resolved by SQL substring scan (never by the retriever). "
        "Recall@k is the fraction of supported cases with ≥1 relevant chunk "
        "in the top k; MRR uses the first relevant rank within the top "
        f"{K_MAX}."
    )
    add("")
    add("## Overall (supported cases)")
    add("")
    add("| Strategy | Recall@5 | Recall@10 | MRR | Zero-result rate |")
    add("|---|---|---|---|---|")
    for s in strategies:
        m = summaries[s]
        add(
            f"| {s} | {_pct(m['recall@5'])} | {_pct(m['recall@10'])} "
            f"| {m['mrr']:.3f} | {_pct(m['zero_result_rate'])} |"
        )
    add("")
    add("## By category (Recall@10 / MRR)")
    add("")
    add("| Category | Cases | " + " | ".join(strategies) + " |")
    add("|---|---|" + "---|" * len(strategies))
    for category in CATEGORIES:
        if category == "unsupported":
            continue
        row_cells = []
        n_cases = None
        for s in strategies:
            cat = summaries[s]["by_category"].get(category)
            if cat is None:
                row_cells.append("—")
            else:
                n_cases = int(cat["cases"])
                row_cells.append(f"{_pct(cat['recall@10'])} / {cat['mrr']:.2f}")
        if n_cases is not None:
            add(f"| {category} | {n_cases} | " + " | ".join(row_cells) + " |")
    add("")
    add("## Unsupported-query behaviour")
    add("")
    add(
        "No relevant evidence exists for these queries; returning fewer "
        "results is better behaviour (a result here is a plausible-looking "
        "non-answer)."
    )
    add("")
    add("| Strategy | Cases | Mean results returned | Cases returning zero |")
    add("|---|---|---|---|")
    for s in strategies:
        u = summaries[s]["unsupported"]
        add(
            f"| {s} | {u['cases']} | {u['mean_results_returned']:.1f} "
            f"| {u['zero_result_cases']} |"
        )
    add("")
    add("## Per-case results (rank of first relevant chunk)")
    add("")
    add(
        "`—` = no relevant chunk in the top "
        f"{K_MAX}; `n/a` = unsupported case (no relevant evidence exists; "
        "the number shown is how many results were returned)."
    )
    add("")
    add("| Case | Category | Truth set | " + " | ".join(strategies) + " |")
    add("|---|---|---|" + "---|" * len(strategies))
    first = by_strategy[strategies[0]]
    for i, base in enumerate(first):
        case = base.case
        cells = []
        for s in strategies:
            r = by_strategy[s][i]
            if r.error:
                cells.append("ERROR")
            elif case.category == "unsupported":
                cells.append(f"n/a ({r.result_count} returned)")
            elif r.first_relevant_rank is None:
                cells.append("—")
            else:
                cells.append(str(r.first_relevant_rank))
        add(
            f"| {case.id} | {case.category} | {base.relevant_count} | "
            + " | ".join(cells)
            + " |"
        )
    add("")
    return "\n".join(lines)


def to_json(
    by_strategy: dict[str, list[CaseResult]],
    summaries: dict[str, dict[str, Any]],
    *,
    run_date: str,
) -> dict[str, Any]:
    """Machine-readable version of the report, with per-case detail."""
    return {
        "run_date": run_date,
        "k_max": K_MAX,
        "summaries": summaries,
        "cases": {
            s: [
                {
                    "id": r.case.id,
                    "category": r.case.category,
                    "query": r.case.query,
                    "filters": r.case.filters,
                    "relevant_count": r.relevant_count,
                    "result_count": r.result_count,
                    "first_relevant_rank": r.first_relevant_rank,
                    "relevant_ranks": r.relevant_ranks,
                    "error": r.error,
                }
                for r in results
            ]
            for s, results in by_strategy.items()
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from datetime import date

    from tools.document_search_tool import RETRIEVAL_STRATEGIES

    parser = argparse.ArgumentParser(
        description="Compare retrieval strategies on the labeled benchmark."
    )
    parser.add_argument(
        "--strategies",
        default=",".join(RETRIEVAL_STRATEGIES),
        help="Comma-separated subset of: " + ", ".join(RETRIEVAL_STRATEGIES),
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write evals/retrieval_report.md and .json (else print only).",
    )
    args = parser.parse_args(argv)

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = set(strategies) - set(RETRIEVAL_STRATEGIES)
    if unknown:
        parser.error(f"unknown strategies: {sorted(unknown)}")

    by_strategy, label_errors = run_benchmark(
        strategies, database_url=args.database_url
    )
    summaries = {s: summarize(rs) for s, rs in by_strategy.items()}

    from db.connection import get_connection

    with get_connection(args.database_url) as conn:
        conn.read_only = True
        stats = conn.execute(
            "SELECT (SELECT count(*) FROM documents) AS docs, "
            "(SELECT count(*) FROM document_chunks) AS chunks"
        ).fetchone()

    from core.config import settings

    run_date = date.today().isoformat()
    report = format_markdown(
        by_strategy,
        summaries,
        run_date=run_date,
        corpus_note=f"{stats['docs']} documents / {stats['chunks']} chunks",
        config_note=(
            f"k={K_MAX}, leg_k={settings.retrieval_leg_k}, "
            f"rrf_k={settings.retrieval_rrf_k}, "
            f"embedder={settings.embedding_model}"
        ),
    )
    print(report)

    if label_errors:
        print("LABEL ERRORS (ground truth drifted from the corpus):")
        for e in label_errors:
            print(f"  - {e}")

    if args.write_report:
        (_REPORT_DIR / "retrieval_report.md").write_text(report)
        (_REPORT_DIR / "retrieval_report.json").write_text(
            json.dumps(
                to_json(by_strategy, summaries, run_date=run_date), indent=2
            )
        )
        print(f"Wrote {_REPORT_DIR / 'retrieval_report.md'} and .json")

    return 1 if label_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
