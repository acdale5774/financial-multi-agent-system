"""Deterministic citation validation for multi-agent answers.

The model writes prose with inline markers like `[S3]` or `[S1, D2]`; the
records behind those IDs exist only in the evidence ledger, minted by code.
This module is the enforcement layer between the model's last message and the
API response:

- HARD RULE (repair, then strip): every marker must resolve to a ledger
  record. Unknown markers trigger one bounded repair turn; whatever is still
  invalid afterwards is stripped and reported in the validation block. The
  endpoint never fails an answer over citations — it degrades honestly.
- WARNING LINTS (reported, never blocking): (a) sentences that state money /
  percent / large-number figures without any marker; (b) marked sentences
  whose numbers don't match any cited value under common display scalings
  (raw, thousands/millions/billions, percent). These are warnings because
  sentence splitting and number matching are heuristics — a hard failure here
  would strip true claims over formatting (e.g. "3.0x higher", a derived
  comparison of two cited figures).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from agents.schemas import DocumentCitation, SimFinCitation, ValidationReport

if TYPE_CHECKING:
    from agents.ledger import EvidenceLedger

# [S1] or [S1, D2, S3] — groups of ledger ids in one bracket.
_MARKER_RE = re.compile(r"\[([SD]\d+(?:\s*,\s*[SD]\d+)*)\]")
# A number, optionally $-prefixed / comma-grouped / %-suffixed.
_NUMBER_RE = re.compile(r"(?<![\w.])[$€£]?(-?\d[\d,]*(?:\.\d+)?)\s*(%?)")
# Signals that a sentence states a figure worth citing.
_NUMERIC_CLAIM_RE = re.compile(
    r"[$€£]\s?\d"
    r"|\d(?:\.\d+)?\s*%"
    r"|\d(?:\.\d+)?\s*(?:trillion|billion|million|[TBM])\b"
    r"|\d{5,}",
)

_SCALES = (1.0, 1e3, 1e6, 1e9, 1e12)
_REL_TOLERANCE = 0.005  # 0.5%: allows display rounding like 184.992 -> "185.0"


def extract_marker_ids(text: str) -> list[str]:
    """All ledger ids referenced by markers, in prose order (with repeats)."""
    ids: list[str] = []
    for group in _MARKER_RE.findall(text):
        ids.extend(part.strip() for part in group.split(","))
    return ids


def split_valid_invalid(text: str, ledger: EvidenceLedger) -> tuple[list[str], list[str]]:
    """Partition marker ids into (valid, invalid) against the ledger."""
    valid: list[str] = []
    invalid: list[str] = []
    for cid in extract_marker_ids(text):
        (valid if ledger.has(cid) else invalid).append(cid)
    return valid, invalid


def strip_invalid_markers(text: str, invalid: set[str]) -> str:
    """Remove invalid ids from markers; drop brackets that become empty."""

    def fix(match: re.Match[str]) -> str:
        kept = [p.strip() for p in match.group(1).split(",") if p.strip() not in invalid]
        return f"[{', '.join(kept)}]" if kept else ""

    cleaned = _MARKER_RE.sub(fix, text)
    return re.sub(r" +([.,;:)])", r"\1", re.sub(r"  +", " ", cleaned)).strip()


def build_repair_message(invalid: list[str], ledger: EvidenceLedger) -> str:
    """One corrective turn: name the bad ids and inventory the real ones."""
    inventory = "\n".join(
        f"- {label}" for label in (_label(c) for c in ledger.citations.values())
    )
    return (
        f"Your answer cites ids that do not exist: {', '.join(sorted(set(invalid)))}. "
        "Rewrite the SAME answer, keeping every figure and claim, but only use "
        "citation ids from this inventory (cite the closest matching record, or "
        "remove the marker if no record supports the claim):\n"
        f"{inventory or '- (no evidence was collected)'}"
    )


def lint_answer(text: str, ledger: EvidenceLedger) -> tuple[int, int, list[str]]:
    """Warning lints: uncited numeric sentences + cited-value mismatches.

    Returns:
        (uncited_numeric_sentences, value_mismatch_sentences, warnings)
    """
    uncited = 0
    mismatched = 0
    warnings: list[str] = []

    for sentence in _sentences(text):
        has_marker = bool(_MARKER_RE.search(sentence))
        if not has_marker:
            if _NUMERIC_CLAIM_RE.search(sentence):
                uncited += 1
            continue

        cited_values = [
            c.value
            for cid in extract_marker_ids(sentence)
            if isinstance((c := ledger.citations.get(cid)), SimFinCitation)
            and c.value is not None
        ]
        numbers = _sentence_numbers(sentence)
        if not cited_values or not numbers:
            continue
        if not any(_matches(n, v) for n in numbers for v in cited_values):
            mismatched += 1
            warnings.append(
                f"No number in {sentence[:80]!r}... matches its cited values "
                "under raw/K/M/B/T/percent scaling."
            )

    return uncited, mismatched, warnings


def apply_validation(
    text: str, ledger: EvidenceLedger, *, repaired: bool
) -> tuple[str, ValidationReport]:
    """Final pass: strip whatever is still invalid, run lints, build report."""
    valid, invalid = split_valid_invalid(text, ledger)
    if invalid:
        text = strip_invalid_markers(text, set(invalid))
    uncited, mismatched, warnings = lint_answer(text, ledger)
    if invalid:
        warnings.insert(
            0,
            f"Stripped citation marker(s) with no ledger record: "
            f"{', '.join(sorted(set(invalid)))}.",
        )
    return text, ValidationReport(
        markers_found=len(valid) + len(invalid),
        invalid_markers_stripped=sorted(set(invalid)),
        repaired=repaired,
        uncited_numeric_sentences=uncited,
        value_mismatch_sentences=mismatched,
        warnings=warnings,
    )


def _sentences(text: str) -> list[str]:
    # Newlines split too, so each markdown table row / bullet lints alone.
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _sentence_numbers(sentence: str) -> list[float]:
    numbers: list[float] = []
    for raw, percent in _NUMBER_RE.findall(sentence):
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if not percent and value == int(value) and (abs(value) <= 31 or 1900 <= value <= 2100):
            continue  # counts ("top 10"), day-like ints, and bare years
        numbers.append(value)
    return numbers


def _matches(number: float, cited: float) -> bool:
    candidates = [cited / s for s in _SCALES] + [cited * 100]  # percent display
    for candidate in candidates:
        if number == candidate:
            return True
        if candidate and abs(number - candidate) / abs(candidate) <= _REL_TOLERANCE:
            return True
    return False


def _label(citation: SimFinCitation | DocumentCitation) -> str:
    if isinstance(citation, SimFinCitation):
        if citation.granularity == "query":
            return f"{citation.id}: query #{citation.sql_index + 1} (whole-result citation)"
        return (
            f"{citation.id}: {citation.ticker or citation.company} "
            f"{citation.metric} {citation.fiscal_period} {citation.fiscal_year} "
            f"= {citation.value} {citation.currency or ''}".rstrip()
        )
    where = f" §{citation.section}" if citation.section else ""
    return f"{citation.id}: {citation.company_name or '?'} {citation.doc_type}{where}"
