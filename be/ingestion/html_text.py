"""HTML → structured text for SEC filings (Workiva HTML) and call transcripts.

Both parsers are built on the stdlib `html.parser` (streaming, no lxml/bs4
dependency — heavy parsers don't install cleanly on this dev machine and the
inputs are large: ~2MB per filing).

Design notes (the chunking strategy depends on these):

SEC filings
    Workiva HTML is <div>/<font> soup with tables used for BOTH layout (cover
    pages) and data (financial statements). We reduce it to a linear list of
    `Block`s — paragraphs and tables — and distinguish data tables from layout
    tables with a shape heuristic (≥2 rows with ≥2 non-empty cells). Layout
    tables are flattened back into text; data tables are linearized row-by-row
    with " | " separators so a row reads as a unit ("Revenue | 2024 | 391,035").

Transcripts
    The transcript HTML is semantically clean: <h6>/`page-header` divs mark
    sections (Management Discussion, Q&A), `speaker` divs carry
    "Name, Title - Question/Answer", and <p> holds the speech. We reduce it to
    `Turn`s — one per speaker utterance — which are the natural semantic unit
    of a call: splitting mid-turn separates a statement from its speaker, and
    merging across turns blurs who said what.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

# ---------------------------------------------------------------------------
# SEC filings
# ---------------------------------------------------------------------------

# Tags that terminate a paragraph accumulation outside tables.
_FLUSH_TAGS = frozenset(
    {"div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "hr", "section"}
)

_ITEM_RE = re.compile(r"(?i)^\s*item\s+(\d{1,2}[A-C]?)\s*[.:—-]*\s*(.{0,120})$")
_PART_RE = re.compile(r"(?i)^\s*part\s+([IVX]{1,4})\s*[.:—-]*\s*(.{0,80})$")


@dataclass
class Block:
    """One linear unit of a filing: a paragraph of text or a linearized table."""

    kind: str  # 'text' | 'table'
    text: str


class _FilingHTMLParser(HTMLParser):
    """Stream Workiva filing HTML into text/table blocks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._buf: list[str] = []
        self._skip_depth = 0  # inside <style>/<script>/<head>
        self._table_depth = 0
        self._rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: list[str] = []
        self._in_cell = False

    # -- tag handling -------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("style", "script", "head"):
            self._skip_depth += 1
            return
        if tag == "table":
            if self._table_depth == 0:
                self._flush_text()
                self._rows = []
            self._table_depth += 1
            return
        if self._table_depth:
            # Only structure the outermost table; nested-table content just
            # flows into the current cell's text.
            if self._table_depth == 1:
                if tag == "tr":
                    self._row = []
                elif tag in ("td", "th"):
                    self._cell = []
                    self._in_cell = True
            return
        if tag in _FLUSH_TAGS:
            self._flush_text()

    def handle_endtag(self, tag: str) -> None:
        if tag in ("style", "script", "head"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "table":
            self._table_depth = max(0, self._table_depth - 1)
            if self._table_depth == 0:
                self._emit_table()
            return
        if self._table_depth:
            if self._table_depth == 1:
                if tag in ("td", "th"):
                    text = " ".join("".join(self._cell).split())
                    self._row.append(text)
                    self._in_cell = False
                elif tag == "tr":
                    if any(c for c in self._row):
                        self._rows.append([c for c in self._row if c])
                    self._row = []
            return
        if tag in _FLUSH_TAGS:
            self._flush_text()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._table_depth:
            if self._in_cell:
                self._cell.append(data)
        else:
            self._buf.append(data)

    # -- emit helpers -------------------------------------------------------

    def _flush_text(self) -> None:
        text = " ".join("".join(self._buf).split())
        self._buf = []
        if text:
            self.blocks.append(Block(kind="text", text=text))

    def _emit_table(self) -> None:
        rows = self._rows
        self._rows = []
        if not rows:
            return
        # Data table: at least two rows that each carry ≥2 non-empty cells.
        multi_cell_rows = sum(1 for r in rows if len(r) >= 2)
        if multi_cell_rows >= 2:
            text = "\n".join(" | ".join(r) for r in rows)
            self.blocks.append(Block(kind="table", text=text))
        else:
            # Layout table (cover pages, signature blocks): flatten to text.
            for r in rows:
                for cell in r:
                    self.blocks.append(Block(kind="text", text=cell))


def parse_filing_html(html: str) -> list[Block]:
    """Parse a Workiva SEC filing into linear text/table blocks."""
    parser = _FilingHTMLParser()
    parser.feed(html)
    parser.close()
    # A trailing paragraph without a closing flush tag:
    parser._flush_text()
    return parser.blocks


def match_section_header(text: str) -> tuple[str | None, str | None]:
    """Detect 'PART II' / 'Item 1A. Risk Factors' style headers.

    Returns (part, item_label): either may be None. Header-like means short —
    a sentence mentioning "Item 1A" mid-paragraph must not switch sections.
    """
    if len(text) > 150:
        return None, None
    m = _PART_RE.match(text)
    if m and len(m.group(2)) <= 60:
        return f"Part {m.group(1).upper()}", None
    m = _ITEM_RE.match(text)
    if m:
        number = m.group(1).upper()
        title = m.group(2).strip().rstrip(".")
        # Table-of-contents rows end with page numbers; body headers don't.
        title = re.sub(r"\s+\d{1,4}$", "", title)
        label = f"Item {number}" + (f". {title}" if title else "")
        return None, label
    return None, None


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------

_SPEAKER_RE = re.compile(
    r"^(?P<name>[^,]+?)"
    r"(?:,\s*(?P<title>.+?))?"
    r"(?:\s*-\s*(?P<qa>Question|Answer))?$"
)


@dataclass
class Turn:
    """One speaker utterance (or preamble/participants line)."""

    section: str  # e.g. 'Management Discussion', 'Q&A', 'Preamble'
    speaker: str | None
    speaker_title: str | None
    qa_role: str | None  # 'Question' | 'Answer' | None
    paragraphs: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)


class _TranscriptHTMLParser(HTMLParser):
    """Stream transcript HTML into speaker turns."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.turns: list[Turn] = []
        self._section = "Preamble"
        self._speaker: tuple[str | None, str | None, str | None] = (None, None, None)
        self._mode: str | None = None  # 'speaker' | 'header' | 'para'
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        cls = dict(attrs).get("class") or ""
        if tag == "div" and "speaker" in cls.split():
            self._mode, self._buf = "speaker", []
        elif tag == "h6":
            self._mode, self._buf = "header", []
        elif tag in ("p", "li"):
            self._mode, self._buf = "para", []

    def handle_endtag(self, tag: str) -> None:
        text = " ".join("".join(self._buf).split())
        if tag == "div" and self._mode == "speaker":
            self._speaker = _parse_speaker(text)
            self._mode = None
        elif tag == "h6" and self._mode == "header":
            if text:
                self._section = _normalize_section(text)
                self._speaker = (None, None, None)
            self._mode = None
        elif tag in ("p", "li") and self._mode == "para":
            if text:
                self._append_paragraph(text)
            self._mode = None

    def handle_data(self, data: str) -> None:
        if self._mode:
            self._buf.append(data)

    def _append_paragraph(self, text: str) -> None:
        name, title, qa = self._speaker
        last = self.turns[-1] if self.turns else None
        if (
            last is not None
            and last.section == self._section
            and last.speaker == name
            and last.qa_role == qa
        ):
            last.paragraphs.append(text)
        else:
            self.turns.append(
                Turn(
                    section=self._section,
                    speaker=name,
                    speaker_title=title,
                    qa_role=qa,
                    paragraphs=[text],
                )
            )


def _parse_speaker(text: str) -> tuple[str | None, str | None, str | None]:
    if not text:
        return None, None, None
    m = _SPEAKER_RE.match(text)
    if not m:
        return text, None, None
    return (
        m.group("name").strip(),
        (m.group("title") or "").strip() or None,
        m.group("qa"),
    )


def _normalize_section(text: str) -> str:
    lowered = text.lower()
    if "management discussion" in lowered:
        return "Management Discussion"
    if lowered in ("q&a", "qa", "q & a") or "q&a" in lowered:
        return "Q&A"
    return text.title() if text.isupper() else text


def parse_transcript_html(html: str) -> list[Turn]:
    """Parse a transcript body into speaker turns (Preamble turns first)."""
    parser = _TranscriptHTMLParser()
    parser.feed(html)
    parser.close()
    return parser.turns
