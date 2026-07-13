"""Deterministic citation validation for multi-agent answers.

The model writes prose with inline markers like `[S3]` or `[S1, D2]` (plus
C#/T# for charts and tables); the records behind those IDs exist only in the
evidence ledger, minted by code.
This module is the enforcement layer between the model's last message and the
API response:

- HARD RULE (repair, then strip): every marker must resolve to a ledger
  record (S/D citations and C charts). Unknown markers trigger one bounded
  repair turn; whatever is still invalid afterwards is stripped and reported.
  Markdown chart images pointing at URLs the ledger never rendered are
  removed. The endpoint never fails an answer over citations — it degrades
  honestly.
- WARNING LINTS (reported, never blocking): (a) sentences that state numeric
  figures without any marker; (b) marked sentences where fewer than half of
  the stated numbers can be found in the cited evidence — matching is
  unit-aware ("$394.3 million" only matches a cited 394.3e9 if the unit says
  billions), and evidence values come from the cited data point, the cited
  query's actual rows (for query-granularity citations), or the cited
  document chunk's text. These stay warnings because sentence splitting and
  number matching are heuristics — hard-failing them would strip true claims
  (e.g. a derived "3.0x higher" comparing two cited figures).

Known residual (documented in agents/README.md): a valid id on a sentence
whose numbers happen to match *other* cited values (swapped attribution)
passes the lints; closing that needs semantic pairing, not string checks.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from agents.schemas import DocumentCitation, SimFinCitation, ValidationReport

if TYPE_CHECKING:
    from agents.ledger import EvidenceLedger

# [S1] / [S1, D2] / [C1] — groups of ledger ids in one bracket. The (?!\()
# lookahead keeps markdown links `[S1](url)` out of citation space.
_MARKER_RE = re.compile(r"\[([SDCT]\d+(?:\s*,\s*[SDCT]\d+)*)\](?!\()")
# Marker-ish syntax the model shouldn't use but sometimes does.
_RANGE_RE = re.compile(r"\[([SDCTsdct])(\d+)\s*[-–]\s*(?:[SDCTsdct])?(\d+)\](?!\()")
_LOOSE_RE = re.compile(r"\[((?:[SDCTsdct]\d+)(?:\s*,\s*(?:[SDCTsdct]\d+))*)\](?!\()")
# Markdown images claiming to be our charts.
_CHART_IMG_RE = re.compile(r"!\[[^\]]*\]\((/charts/[^)]+)\)")

# A number with optional sign (either side of the currency symbol), currency
# prefix, comma grouping, and unit suffix.
_NUMBER_RE = re.compile(
    r"(?<![\w.])(-?)[$€£]?(-?\d[\d,]*(?:\.\d+)?)"
    r"\s*(%|(?i:trillion|billion|million|thousand|bn|mn|tn)\b|[KMBTkmbt](?![A-Za-z0-9]))?"
)
# Signals that a sentence states a figure worth citing.
_NUMERIC_CLAIM_RE = re.compile(
    r"[$€£]\s?\d"
    r"|\d(?:\.\d+)?\s*%"
    r"|(?i:\d(?:\.\d+)?\s*(?:trillion|billion|million|thousand|bn|mn|tn)\b)"
    r"|\d(?:\.\d+)?\s*[KMBT]\b"
    r"|\d{5,}"
    r"|\d{1,3}(?:,\d{3})+"
    r"|\d+\.\d+"
)

_UNIT_SCALE = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "t": 1e12,
    "tn": 1e12,
    "trillion": 1e12,
}
_SCALES = (1.0, 1e3, 1e6, 1e9, 1e12)
_REL_TOLERANCE = 0.005  # 0.5%: allows display rounding like 184.992 -> "185.0"


def normalize_markers(text: str) -> str:
    """Canonicalize marker-ish syntax so validation sees it.

    `[s2]` -> `[S2]`; `[S1-S4]` -> `[S1, S2, S3, S4]` (so a range citing
    nonexistent ids gets caught instead of shipping unvalidated).
    """

    def expand(match: re.Match[str]) -> str:
        prefix = match.group(1).upper()
        start, end = int(match.group(2)), int(match.group(3))
        if end < start or end - start > 19:
            return match.group(0)
        return "[" + ", ".join(f"{prefix}{i}" for i in range(start, end + 1)) + "]"

    text = _RANGE_RE.sub(expand, text)
    return _LOOSE_RE.sub(lambda m: "[" + m.group(1).upper() + "]", text)


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
    labels = [_label(c) for c in ledger.citations.values()]
    labels += [f"{c.id}: chart {c.spec.get('title', '')!r} at {c.url}" for c in
               ledger.charts.values()]
    inventory = "\n".join(f"- {label}" for label in labels)
    return (
        f"Your answer cites ids that do not exist: {', '.join(sorted(set(invalid)))}. "
        "Rewrite the SAME answer, keeping every figure and claim, but only use "
        "citation ids from this inventory (cite the closest matching record, or "
        "remove the marker if no record supports the claim):\n"
        f"{inventory or '- (no evidence was collected)'}"
    )


def parse_numbers(text: str) -> list[tuple[float, str, int]]:
    """(value, unit, decimals) triples found in text.

    `unit` is '', '%', or a scale word; `decimals` is how many decimal places
    the text displayed — matching uses it to allow display quantization
    ("1.8B" legitimately stands for anything in [1.75e9, 1.85e9]).
    """
    out: list[tuple[float, str, int]] = []
    for sign, raw, unit in _NUMBER_RE.findall(text):
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if sign == "-":
            value = -abs(value)
        decimals = len(raw.rsplit(".", 1)[1]) if "." in raw else 0
        out.append((value, (unit or "").lower(), decimals))
    return out


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

        numbers = _claim_numbers(sentence)
        if not numbers:
            continue
        candidates: list[float] = []
        for cid in extract_marker_ids(sentence):
            candidates.extend(ledger.values_for(cid))
        if not candidates:
            # Markers that carry no checkable values (e.g. chart-only ids)
            # leave the figures effectively unbacked.
            uncited += 1
            continue
        matched = sum(1 for n, u, d in numbers if _matches(n, u, d, candidates))
        if matched * 2 < len(numbers):
            mismatched += 1
            warnings.append(
                f"Only {matched}/{len(numbers)} figures in {sentence[:80]!r}... "
                "were found in the cited evidence (unit-aware match)."
            )

    return uncited, mismatched, warnings


def apply_validation(
    text: str, ledger: EvidenceLedger, *, repaired: bool
) -> tuple[str, ValidationReport]:
    """Final pass: strip whatever is still invalid, run lints, build report."""
    text = normalize_markers(text)
    warnings: list[str] = []

    valid, invalid = split_valid_invalid(text, ledger)
    if invalid:
        text = strip_invalid_markers(text, set(invalid))
        warnings.append(
            "Stripped citation marker(s) with no ledger record: "
            f"{', '.join(sorted(set(invalid)))}."
        )

    real_urls = {c.url for c in ledger.charts.values()}
    phantom = [u for u in _CHART_IMG_RE.findall(text) if u not in real_urls]
    for url in phantom:
        text = re.sub(r"!\[[^\]]*\]\(" + re.escape(url) + r"\)", "", text).strip()
        warnings.append(f"Removed chart image that was never rendered: {url}.")

    uncited, mismatched, lint_warnings = lint_answer(text, ledger)
    warnings.extend(lint_warnings)
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


def _claim_numbers(sentence: str) -> list[tuple[float, str, int]]:
    """Numbers worth checking: skips bare years and small unitless counts."""
    out = []
    for value, unit, decimals in parse_numbers(sentence):
        if unit == "" and value == int(value) and (abs(value) <= 31 or 1900 <= value <= 2100):
            continue  # counts ("top 10"), day-like ints, and bare years
        out.append((value, unit, decimals))
    return out


def _matches(number: float, unit: str, decimals: int, cited: list[float]) -> bool:
    for value in cited:
        if unit == "%":
            if _close(number, value * 100, decimals) or _close(number, value, decimals):
                return True
        elif unit in _UNIT_SCALE:
            # Unit stated in prose pins the magnitude: "$394.3 million" only
            # matches evidence at ~394.3e6 (or a pre-scaled column at 394.3).
            if _close(number * _UNIT_SCALE[unit], value, decimals, _UNIT_SCALE[unit]) or _close(
                number, value, decimals
            ):
                return True
        else:
            if any(_close(number, value / s, decimals) for s in _SCALES):
                return True
    return False


def _close(a: float, b: float, decimals: int = 0, scale: float = 1.0) -> bool:
    """True when `a` could be `b` displayed at this precision.

    Display quantization dominates for short mantissas ("1.8B" is anything
    in [1.75e9, 1.85e9] — a 2.8% band); the relative tolerance covers
    high-precision displays.
    """
    if a == b:
        return True
    quantum = 0.5 * 10 ** (-decimals) * scale
    if abs(a - b) <= quantum * 1.0000001:
        return True
    return b != 0 and abs(a - b) / abs(b) <= _REL_TOLERANCE


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
