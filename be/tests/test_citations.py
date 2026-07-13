"""Tests for the deterministic citation validator (agents/citations.py)."""

from __future__ import annotations

from agents import citations
from agents.ledger import EvidenceLedger


def _ledger_with_revenue() -> tuple[EvidenceLedger, str]:
    ledger = EvidenceLedger()
    _, cite_maps, _ = ledger.record_sql(
        "SELECT ...",
        [
            {
                "ticker": "F",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "revenue": 184992000000.0,
            }
        ],
    )
    return ledger, cite_maps[0]["revenue"]


def test_extract_marker_ids_handles_groups():
    assert citations.extract_marker_ids("A [S1] b [S2, D3] c [S1]") == ["S1", "S2", "D3", "S1"]


def test_strip_invalid_keeps_valid_part_of_group():
    text = "Revenue rose [S1, S9] and fell [S9]."
    assert citations.strip_invalid_markers(text, {"S9"}) == "Revenue rose [S1] and fell."


def test_repair_message_lists_inventory():
    ledger, cid = _ledger_with_revenue()
    message = citations.build_repair_message(["S9"], ledger)
    assert "S9" in message
    assert f"{cid}: F revenue FY 2024" in message


def test_lint_flags_uncited_numeric_sentence():
    ledger, _ = _ledger_with_revenue()
    uncited, mismatched, _ = citations.lint_answer(
        "Revenue was $185.0B in FY2024. The top 10 list follows.", ledger
    )
    assert uncited == 1  # the $185.0B sentence
    assert mismatched == 0  # 'top 10' and 'FY2024' are exempt, not mismatches


def test_lint_accepts_rounded_billion_display():
    ledger, cid = _ledger_with_revenue()
    uncited, mismatched, _ = citations.lint_answer(
        f"Ford revenue was $185.0B in FY2024 [{cid}].", ledger
    )
    assert (uncited, mismatched) == (0, 0)


def test_lint_flags_wrong_value_next_to_marker():
    ledger, cid = _ledger_with_revenue()
    _, mismatched, warnings = citations.lint_answer(
        f"Ford revenue was $210.0B in FY2024 [{cid}].", ledger
    )
    assert mismatched == 1
    assert warnings


def test_lint_accepts_percent_display_of_ratio():
    ledger = EvidenceLedger()
    _, cite_maps, _ = ledger.record_sql(
        "SELECT ...",
        [{"ticker": "F", "fiscal_year": 2024, "fiscal_period": "FY", "growth": 0.125}],
    )
    cid = cite_maps[0]["growth"]
    _, mismatched, _ = citations.lint_answer(f"Revenue grew 12.5% [{cid}].", ledger)
    assert mismatched == 0


def test_apply_validation_strips_and_reports():
    ledger, cid = _ledger_with_revenue()
    text, report = citations.apply_validation(
        f"Revenue was $185.0B [{cid}]. Margin was 8% [S9].", ledger, repaired=True
    )
    assert "[S9]" not in text
    assert f"[{cid}]" in text
    assert report.invalid_markers_stripped == ["S9"]
    assert report.repaired is True
    assert report.markers_found == 2
    # After stripping S9 its sentence still states 8% uncited.
    assert report.uncited_numeric_sentences == 1
