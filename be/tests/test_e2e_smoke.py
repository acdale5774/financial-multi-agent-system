"""End-to-end smoke test: the real pipeline against real infrastructure.

Every *other* test in this suite is deterministic — fake chat models drive the
real graphs and DB access is monkeypatched at the tool boundary (see
`tests/README.md`). That proves application *logic* (marker validation, ledger
minting, chart hydration, SQL guards, the API contract). It cannot prove the
three things that only emerge against real infrastructure:

  * that the live model selects the right tools and routes correctly,
  * that retrieval returns relevant chunks, and
  * that the provenance guarantee holds on prose the model *actually* generated
    (mocked prose is authored by the test, so it proves nothing about the model).

This module is the only test that runs `answer_multi()` end to end against a
live Postgres AND a live OpenAI model. Because it makes real (paid) model calls,
it is **opt-in**: the default `pytest` run always skips it, so unit runs stay
offline, free, and dependency-free even when a key and DB happen to be present
(the same philosophy as the eval harness's `--agent` flag).

Run it explicitly with the stack up and a key set:

    RUN_E2E=1 pytest -m e2e               # Postgres from ../infra must be running
                                          # and OPENAI_API_KEY set in the env

It asserts *structural* invariants, never exact figures — a reasoning model's
wording drifts, so pinning numbers would make it flaky without adding signal.
The invariants it checks (end-to-end provenance; chart-to-source consistency;
route behaviour) are precisely the properties a mocked test cannot establish.
"""

from __future__ import annotations

import os

import pytest

from agents.citations import extract_marker_ids, normalize_markers

VALID_ROUTES = {"quant", "docs", "hybrid", "unsupported"}


def _database_available() -> bool:
    try:
        from tools.sql_tool import run_read_only_sql

        run_read_only_sql("SELECT 1 AS answer", statement_timeout_ms=3000)
        return True
    except Exception:  # noqa: BLE001 - any failure means "no live DB for this test"
        return False


def _model_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _e2e_opt_in() -> bool:
    # Opt-in gate so a plain `pytest` never makes paid model calls, even with a
    # key and DB present. Run with `RUN_E2E=1 pytest -m e2e`.
    return os.getenv("RUN_E2E", "").strip().lower() in {"1", "true", "yes", "on"}


# Full pipeline needs both a live model and a live DB; routing alone needs only
# the model (the `unsupported` path never touches the database). Both are also
# gated on the RUN_E2E opt-in.
requires_stack = pytest.mark.skipif(
    not (_e2e_opt_in() and _database_available() and _model_available()),
    reason="set RUN_E2E=1 with a live Postgres and OPENAI_API_KEY to run",
)
requires_model = pytest.mark.skipif(
    not (_e2e_opt_in() and _model_available()),
    reason="set RUN_E2E=1 with OPENAI_API_KEY to run (no database required)",
)


@pytest.mark.e2e
@requires_model
def test_offtopic_question_routes_unsupported_without_a_specialist_run() -> None:
    """The real dispatcher classifies an off-topic question as `unsupported`
    and short-circuits to a canned capabilities answer — no SQL, no specialist,
    no citations. Cheap real-model coverage of the routing decision itself
    (one dispatch call, no DB), and the `unsupported`-behaviour eval dimension.
    """
    from agents.multi_agent import answer_multi

    ans = answer_multi("What's a good recipe for sourdough bread?")

    assert ans.route == "unsupported", (
        f"off-topic question should route to unsupported, got {ans.route!r} "
        f"(reason: {ans.route_reason})"
    )
    assert ans.answer.strip(), "unsupported route should still return a capabilities answer"
    assert ans.sql_queries == [], "unsupported route must not spend a specialist/SQL run"
    assert ans.citations == [], "a canned capabilities answer cites nothing"


@pytest.mark.e2e
@requires_stack
def test_quant_answer_is_fully_grounded_end_to_end() -> None:
    """A real quant question, answered by the real model over the real DB.

    Asserts the invariants that no mocked test can reach:
      1. it routes to a specialist that can query + visualise,
      2. it actually queried the structured store and produced evidence,
      3. PROVENANCE: every citation marker the model rendered in the final
         prose resolves to a real evidence record (validation has already
         stripped danglers, so whatever survives *must* resolve), and
      4. CHART-TO-SOURCE: every id a chart was hydrated from is a real
         citation on this same answer (requirement 4's traceability).
    """
    from agents.multi_agent import answer_multi

    ans = answer_multi("Plot the revenue for Ford over the last 3 quarters.")

    # 1. Routed to a specialist able to query + chart.
    assert ans.route in {"quant", "hybrid"}, (
        f"a revenue-plot question should route quant/hybrid, got {ans.route!r} "
        f"(reason: {ans.route_reason})"
    )
    assert ans.route in VALID_ROUTES
    assert ans.answer.strip()

    # 2. It used the structured store and produced citable evidence.
    assert ans.sql_queries, "a revenue question should execute at least one SQL query"
    assert ans.citations, "an answer stating figures should carry citations"

    # 3. The end-to-end provenance invariant: every marker in the model's own
    #    prose is backed by a real record. This is the claim a mocked test
    #    cannot make, because in a mocked test the prose is authored by us.
    record_ids = {c.id for c in ans.citations}
    record_ids |= {ch.id for ch in ans.charts}
    record_ids |= {t.id for t in ans.tables}
    rendered = set(extract_marker_ids(normalize_markers(ans.answer)))
    dangling = rendered - record_ids
    assert not dangling, (
        f"answer renders citation markers with no backing record: {sorted(dangling)}"
    )

    # 4. An explicit "plot" request should yield a traceable visual, and every
    #    value it plotted must trace to a cited data point.
    assert ans.charts or ans.tables, "an explicit plot request should produce a chart or table"
    for chart in ans.charts:
        unknown = set(chart.citation_ids) - record_ids
        assert not unknown, f"chart {chart.id} was hydrated from unknown records: {sorted(unknown)}"
