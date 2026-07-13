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


# --- validator v2: loopholes closed after adversarial review -----------------


def test_magnitude_word_pins_the_scale():
    ledger, cid = _ledger_with_revenue()
    _, mismatched, _ = citations.lint_answer(
        f"Ford revenue was $185.0 million in FY2024 [{cid}].", ledger
    )
    assert mismatched == 1  # 185.0e6 != 184.992e9; blind scaling no longer saves it


def test_document_citation_numbers_checked_against_chunk_text():
    from tools.document_search_tool import SearchResult

    ledger = EvidenceLedger()
    chunk = SearchResult(
        document_id="octus-1",
        chunk_index=0,
        text="We generated $1.4 billion of free cash flow this quarter.",
        score=0.7,
        title="Delta transcript",
        metadata={"doc_type": "Transcript", "company_name": "Delta Air Lines"},
    )
    (cite,) = ledger.record_documents("fcf", None, [chunk])

    _, ok_mismatch, _ = citations.lint_answer(
        f"Delta generated $1.4 billion of free cash flow [{cite.id}].", ledger
    )
    _, bad_mismatch, _ = citations.lint_answer(
        f"Delta revenue reached $500 billion [{cite.id}].", ledger
    )
    assert ok_mismatch == 0
    assert bad_mismatch == 1


def test_query_granularity_citation_checks_against_recorded_rows():
    ledger = EvidenceLedger()
    _, cite_maps, _ = ledger.record_sql(
        "SELECT sum(value) AS total ...", [{"total": 61643000000.0}]
    )
    cid = cite_maps[0]["*"]

    _, ok, _ = citations.lint_answer(f"Total revenue was $61.6B [{cid}].", ledger)
    _, bad, _ = citations.lint_answer(f"Total revenue was $999 trillion [{cid}].", ledger)
    assert ok == 0
    assert bad == 1


def test_fabricated_riders_fail_the_half_rule():
    ledger, cid = _ledger_with_revenue()
    _, mismatched, _ = citations.lint_answer(
        f"Revenue was $185.0B [{cid}]; margins hit 25.3%, headcount rose to 164,000, "
        "and R&D reached $88B.",
        ledger,
    )
    assert mismatched == 1  # only 1 of 4 figures found in evidence


def test_invalid_chart_marker_stripped_and_valid_one_kept():
    ledger, cid = _ledger_with_revenue()
    ledger.record_chart(
        {"url": "/charts/x.png", "file": "/tmp/x.png", "spec": {"title": "t"}}, [cid]
    )

    text, report = citations.apply_validation(
        "See the trend [C1], not [C9].", ledger, repaired=False
    )
    assert "[C1]" in text and "[C9]" not in text
    assert report.invalid_markers_stripped == ["C9"]


def test_phantom_chart_image_removed_real_one_kept():
    ledger, cid = _ledger_with_revenue()
    ledger.record_chart({"url": "/charts/real.png", "file": "/tmp/r.png", "spec": {}}, [cid])

    text, report = citations.apply_validation(
        "![real](/charts/real.png)\n![fake](/charts/deadbeef.png)", ledger, repaired=False
    )
    assert "/charts/real.png" in text
    assert "deadbeef" not in text
    assert any("never rendered" in w for w in report.warnings)


def test_range_and_lowercase_markers_are_normalized_then_validated():
    ledger, cid = _ledger_with_revenue()  # only S1 exists
    text, report = citations.apply_validation(
        "Revenue rose across the period [S1-S3]. It was strong [s1].", ledger, repaired=False
    )
    assert report.invalid_markers_stripped == ["S2", "S3"]
    assert text.count("[S1]") == 2


def test_comma_ints_and_bare_decimals_count_as_numeric_claims():
    ledger, _ = _ledger_with_revenue()
    uncited, _, _ = citations.lint_answer(
        "Diluted shares were 16,325,819. EPS came in at 6.11.", ledger
    )
    assert uncited == 2


def test_markdown_link_is_not_a_citation_marker():
    assert citations.extract_marker_ids("See [S1](https://example.com) here") == []


def test_display_rounding_of_generated_table_cells_is_accepted():
    ledger = EvidenceLedger()
    _, cite_maps, _ = ledger.record_sql(
        "SELECT ...",
        [
            {
                "ticker": "F",
                "fiscal_year": 2024,
                "fiscal_period": "Q4",
                "net_income": 1831000000.0,
                "eps": 0.4534,
            }
        ],
    )
    cites = cite_maps[0]
    _, mismatched, warnings = citations.lint_answer(
        f"| FY2024 Q4 | 1.8B [{cites['net_income']}] | 0.45 [{cites['eps']}] |", ledger
    )
    assert mismatched == 0, warnings


def test_quantization_does_not_excuse_wrong_magnitude():
    ledger, cid = _ledger_with_revenue()
    _, mismatched, _ = citations.lint_answer(
        f"Revenue was $185.0 million [{cid}].", ledger
    )
    assert mismatched == 1


def test_negative_value_with_minus_before_currency_symbol_matches():
    ledger = EvidenceLedger()
    _, cite_maps, _ = ledger.record_sql(
        "SELECT ...",
        [{"ticker": "AAMC", "fiscal_year": 2023, "fiscal_period": "FY", "revenue": -17320000.0}],
    )
    cid = cite_maps[0]["revenue"]
    _, mismatched, warnings = citations.lint_answer(
        f"| AAMC | -$17.32M FY2023 USD [{cid}] |", ledger
    )
    assert mismatched == 0, warnings
