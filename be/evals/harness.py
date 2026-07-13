"""Runner + scorecard for the text-to-SQL evaluation set.

`run_ground_truth` needs only Postgres; `run_agent` additionally needs the
structured-query agent (OpenAI). Both grade against the same measured answers in
`evals.cases`, so a category the agent misses is a semantic the curated
tool-notes don't reliably convey — the prioritization signal for a governed
semantic layer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from evals.cases import CASES, EvalCase

# Agent-issued and reference queries hit a 3M-row table; give them headroom.
_SQL_TIMEOUT_MS = 15_000


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> float | None:
    """Coerce a psycopg cell (Decimal/int/float/None) to float."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rel_close(got: float, expected: float, rel_tolerance: float) -> bool:
    """True if `got` is within `rel_tolerance` (relative) of `expected`."""
    if expected == 0:
        return abs(got) <= 1e-9
    return abs(got - expected) / abs(expected) <= rel_tolerance


def _fetch_answer(sql: str, database_url: str | None) -> float | None:
    """Run a reference/trap query and return its single `answer` value (or None)."""
    from tools.sql_tool import run_read_only_sql

    rows = run_read_only_sql(
        sql, max_rows=5, statement_timeout_ms=_SQL_TIMEOUT_MS, database_url=database_url
    )
    if not rows:
        return None
    return _to_float(rows[0].get("answer"))


def _value_present(expected: float, unit: str, text: str) -> bool:
    """Whether `expected` appears in prose, unit/scale/rounding-aware.

    Reuses the citation validator's number parser and matcher (the same logic
    that checks whether a cited figure backs a sentence) so grading matches how
    the system already reasons about numbers. Comparison is magnitude-based:
    "declined 2.8%", "-2.8%", "$61.6 billion", and "61,555,000,000" all count.
    """
    from agents.citations import _matches, parse_numbers

    target = abs(expected)
    for value, parsed_unit, decimals in parse_numbers(text):
        if _matches(abs(value), parsed_unit or unit, decimals, [target]):
            return True
    return False


# ---------------------------------------------------------------------------
# Ground-truth layer (no API key)
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthResult:
    """Outcome of running one case's golden SQL against the live database."""

    case: EvalCase
    passed: bool
    got: float | None
    trap_got: float | None = None
    error: str | None = None

    @property
    def trap_reproduced(self) -> bool:
        """True if the naive query still produces the documented wrong answer."""
        if self.case.trap_sql is None:
            return False
        if self.case.trap_value is None:
            return self.trap_got is None  # trap = "returns no rows"
        return self.trap_got is not None and _rel_close(
            self.trap_got, self.case.trap_value, max(self.case.rel_tolerance, 0.005)
        )


def run_ground_truth(
    cases: list[EvalCase] | None = None, *, database_url: str | None = None
) -> list[GroundTruthResult]:
    """Execute every case's golden SQL and check it against the measured answer."""
    results: list[GroundTruthResult] = []
    for case in cases or CASES:
        try:
            got = _fetch_answer(case.reference_sql, database_url)
            trap_got = (
                _fetch_answer(case.trap_sql, database_url) if case.trap_sql else None
            )
        except Exception as exc:  # noqa: BLE001 - surfaced per-case, never aborts the run
            results.append(
                GroundTruthResult(
                    case=case, passed=False, got=None, error=f"{type(exc).__name__}: {exc}"
                )
            )
            continue
        passed = got is not None and _rel_close(got, case.expected_value, case.rel_tolerance)
        results.append(GroundTruthResult(case=case, passed=passed, got=got, trap_got=trap_got))
    return results


# ---------------------------------------------------------------------------
# Agent layer (opt-in; needs the structured-query agent + OpenAI key)
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """Outcome of running the real agent on one case's natural-language question."""

    case: EvalCase
    passed: bool
    answer_present: bool
    trap_present: bool
    answer_excerpt: str
    sql_ran: list[str] = field(default_factory=list)
    iterations: int = 0
    error: str | None = None


def run_agent(
    cases: list[EvalCase] | None = None,
    *,
    database_url: str | None = None,
    model: str | None = None,
) -> list[AgentResult]:
    """Run the structured-query agent per case and grade its prose answer.

    A case passes when the correct value is present in the answer. If the trap
    value is *also* present, the case still counts as failed-with-warning (the
    agent surfaced the semantics-blind figure), captured via `trap_present`.
    """
    from agents.orchestrator import answer as agent_answer

    results: list[AgentResult] = []
    for case in cases or CASES:
        try:
            result = agent_answer(case.question, model=model, database_url=database_url)
        except Exception as exc:  # noqa: BLE001 - one bad case shouldn't kill the sweep
            results.append(
                AgentResult(
                    case=case,
                    passed=False,
                    answer_present=False,
                    trap_present=False,
                    answer_excerpt="",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        present = _value_present(case.expected_value, case.unit, result.answer)
        trap_present = (
            case.trap_value is not None
            and _value_present(case.trap_value, case.unit, result.answer)
        )
        results.append(
            AgentResult(
                case=case,
                passed=present and not trap_present,
                answer_present=present,
                trap_present=trap_present,
                answer_excerpt=" ".join(result.answer.split())[:280],
                sql_ran=[q.query.strip() for q in result.sql_queries],
                iterations=result.iterations,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _humanize(value: float | None, unit: str) -> str:
    """Compact human display: 383_285_000_000 -> '$383.29B', 6.1341 -> '6.13'."""
    if value is None:
        return "∅ (no rows)"
    if unit == "%":
        return f"{value:.2f}%"
    if unit == "ratio":
        return f"{value:.2f}"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= scale:
            return f"{sign}${magnitude / scale:.2f}{suffix}"
    return f"{sign}${magnitude:.2f}"


def _category_summary(pairs: list[tuple[str, bool]]) -> list[str]:
    """Per-category pass tallies, in first-seen order."""
    order: list[str] = []
    tally: dict[str, list[int]] = {}
    for category, passed in pairs:
        if category not in tally:
            tally[category] = [0, 0]
            order.append(category)
        tally[category][0] += int(passed)
        tally[category][1] += 1
    return [f"  {cat:<20} {tally[cat][0]}/{tally[cat][1]}" for cat in order]


def format_report(
    ground: list[GroundTruthResult], agent: list[AgentResult] | None = None
) -> str:
    """Render a human scorecard for a run."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("GROUND TRUTH  (golden SQL via the read-only tool — no LLM)")
    lines.append("=" * 72)
    for r in ground:
        mark = "PASS" if r.passed else "FAIL"
        detail = f"got {_humanize(r.got, r.case.unit)}"
        if not r.passed:
            detail = (
                f"got {_humanize(r.got, r.case.unit)} "
                f"!= expected {_humanize(r.case.expected_value, r.case.unit)}"
            )
            if r.error:
                detail = r.error
        lines.append(f"[{mark}] {r.case.category:<18} {r.case.id}")
        lines.append(f"       {detail}")
        if r.case.trap_sql:
            status = "reproduced" if r.trap_reproduced else "NOT reproduced"
            lines.append(
                f"       trap ({r.case.trap_label}): "
                f"{_humanize(r.trap_got, r.case.unit)} [{status}]"
            )
    gt_pass = sum(r.passed for r in ground)
    lines.append("")
    lines.append(f"Ground truth: {gt_pass}/{len(ground)} passed")
    lines.extend(_category_summary([(r.case.category, r.passed) for r in ground]))

    if agent is not None:
        lines.append("")
        lines.append("=" * 72)
        lines.append("AGENT GRADING  (real NL -> SQL agent, graded vs ground truth)")
        lines.append("=" * 72)
        for r in agent:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"[{mark}] {r.case.category:<18} {r.case.id}  ({r.iterations} iters)")
            if r.error:
                lines.append(f"       ERROR: {r.error}")
                continue
            flags = []
            if not r.answer_present:
                flags.append("correct value absent")
            if r.trap_present:
                flags.append("TRAP value present")
            if flags:
                lines.append(f"       {', '.join(flags)}")
            lines.append(f"       answer: {r.answer_excerpt}")
        ag_pass = sum(r.passed for r in agent)
        lines.append("")
        lines.append(f"Agent: {ag_pass}/{len(agent)} passed")
        lines.extend(_category_summary([(r.case.category, r.passed) for r in agent]))
        lines.append("")
        lines.append(
            "Failing categories above are the semantics the prose tool-notes do "
            "not reliably\nconvey — i.e. the ones to promote into a governed, "
            "machine-readable model first."
        )

    return "\n".join(lines)


def _as_dict(ground: list[GroundTruthResult], agent: list[AgentResult] | None) -> dict[str, Any]:
    """JSON-serializable view of a run."""
    out: dict[str, Any] = {
        "ground_truth": [
            {
                "id": r.case.id,
                "category": r.case.category,
                "passed": r.passed,
                "got": r.got,
                "expected": r.case.expected_value,
                "semantic_rule": r.case.semantic_rule,
                "trap_reproduced": r.trap_reproduced,
                "error": r.error,
            }
            for r in ground
        ]
    }
    if agent is not None:
        out["agent"] = [
            {
                "id": r.case.id,
                "category": r.case.category,
                "passed": r.passed,
                "answer_present": r.answer_present,
                "trap_present": r.trap_present,
                "sql_ran": r.sql_ran,
                "answer_excerpt": r.answer_excerpt,
                "error": r.error,
            }
            for r in agent
        ]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the text-to-SQL evaluation set (ground truth + optional agent grading)."
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Also run the real NL->SQL agent (needs OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Only run cases in this semantic category (e.g. stock-aggregation).",
    )
    parser.add_argument("--model", default=None, help="Override the agent model.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of the scorecard.")
    args = parser.parse_args(argv)

    cases = CASES
    if args.category:
        cases = [c for c in CASES if c.category == args.category]
        if not cases:
            parser.error(
                f"No cases in category '{args.category}'. "
                f"Available: {', '.join(sorted({c.category for c in CASES}))}."
            )

    ground = run_ground_truth(cases)
    agent = run_agent(cases, model=args.model) if args.agent else None

    if args.json:
        print(json.dumps(_as_dict(ground, agent), indent=2, default=str))
    else:
        print(format_report(ground, agent))

    # Non-zero exit if ground truth regresses (agent misses are informational).
    return 0 if all(r.passed for r in ground) else 1


if __name__ == "__main__":
    raise SystemExit(main())
