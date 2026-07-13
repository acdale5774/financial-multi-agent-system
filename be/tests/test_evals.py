"""Tests for the text-to-SQL evaluation harness.

Two tiers:
- Pure tests (no DB): the case set is well-formed, and the numeric grading
  helper matches values the way the citation validator does.
- Integration test (live DB): every golden query returns its measured answer.
  Skipped automatically when Postgres isn't reachable, so the default unit run
  stays dependency-free.
"""

from __future__ import annotations

import pytest

from evals.cases import CASES
from evals.harness import _value_present, run_ground_truth


def _database_available() -> bool:
    try:
        from tools.sql_tool import run_read_only_sql

        run_read_only_sql("SELECT 1 AS answer", statement_timeout_ms=3000)
        return True
    except Exception:  # noqa: BLE001 - any failure means "no DB for this test"
        return False


# --- Pure tests -----------------------------------------------------------


def test_case_ids_unique() -> None:
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_cases_well_formed(case) -> None:
    assert "select" in case.reference_sql.lower()
    assert "answer" in case.reference_sql.lower()  # returns the conventional column
    assert case.semantic_rule  # every case names the rule it guards
    assert case.question.strip()
    # A declared trap must carry a label so the report can explain it.
    if case.trap_sql is not None:
        assert case.trap_label


def test_value_present_is_scale_and_sign_aware() -> None:
    # Same magnitude expressed different ways all count as present.
    assert _value_present(61_555_000_000, "USD", "ended the year with $61.6 billion in cash")
    assert _value_present(61_555_000_000, "USD", "cash was 61,555,000,000")
    # Magnitude-based: a decline stated without a minus sign still matches.
    assert _value_present(-2.80, "%", "revenue declined 2.8% year over year")
    # A clearly different magnitude does not match.
    assert not _value_present(61_555_000_000, "USD", "revenue was $383.3 billion")


# --- Integration test (live DB) ------------------------------------------


@pytest.mark.skipif(not _database_available(), reason="Postgres not reachable")
def test_ground_truth_all_pass() -> None:
    results = run_ground_truth()
    failures = [
        f"{r.case.id}: got {r.got!r} != expected {r.case.expected_value!r} ({r.error or ''})"
        for r in results
        if not r.passed
    ]
    assert not failures, "Golden SQL drifted from measured answers:\n" + "\n".join(failures)


@pytest.mark.skipif(not _database_available(), reason="Postgres not reachable")
def test_traps_reproduce() -> None:
    # The documented naive queries should still yield their wrong answers —
    # otherwise the trap no longer illustrates the failure mode.
    for r in run_ground_truth():
        if r.case.trap_sql is not None:
            assert r.trap_reproduced, f"{r.case.id}: trap no longer reproduces"
