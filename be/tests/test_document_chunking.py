"""Unit tests for HTML parsing + chunking (no network or database required)."""

from __future__ import annotations

from datetime import date

from ingestion.document_ingest import SourceDocument, chunk_filing, chunk_transcript
from ingestion.html_text import match_section_header, parse_filing_html, parse_transcript_html

COMPANY = {
    "octus_company_id": "742214",
    "company_name": "Delta Air Lines",
    "sub_industry": "Passenger Airlines",
}


def _filing(html: str) -> SourceDocument:
    return SourceDocument(
        external_id="doc-1",
        source_type="SEC Filing",
        doc_type="10-K",
        document_date=date(2024, 2, 1),
        company=COMPANY,
        title=None,
        html=html,
    )


def _transcript(html: str) -> SourceDocument:
    return SourceDocument(
        external_id="doc-2",
        source_type="Transcript",
        doc_type="Transcript",
        document_date=date(2024, 9, 12),
        company=COMPANY,
        title=None,
        html=html,
    )


# ---------------------------------------------------------------------------
# Filing parsing
# ---------------------------------------------------------------------------

FILING_HTML = """
<html><body>
<div><font>PART I</font></div>
<div><font>Item 1A. Risk Factors</font></div>
<div><font>Fuel price volatility could materially affect our results.</font></div>
<div><font>We depend on regional airports for a large share of traffic.</font></div>
<div><font>Item 7. Management's Discussion and Analysis</font></div>
<table>
  <tr><td>Metric</td><td>2024</td><td>2023</td></tr>
  <tr><td>Revenue</td><td>61,643</td><td>58,048</td></tr>
  <tr><td>Net income</td><td>3,457</td><td>4,609</td></tr>
</table>
<div><font>Revenue grew on strong demand.</font></div>
</body></html>
"""


def test_filing_blocks_and_tables() -> None:
    blocks = parse_filing_html(FILING_HTML)
    kinds = [b.kind for b in blocks]
    assert "table" in kinds
    table = next(b for b in blocks if b.kind == "table")
    # Rows linearized with ' | ' so label↔value alignment survives.
    assert "Revenue | 61,643 | 58,048" in table.text


def test_layout_table_is_flattened_to_text() -> None:
    html = """<table>
      <tr><td>UNITED STATES</td></tr>
      <tr><td>SECURITIES AND EXCHANGE COMMISSION</td></tr>
    </table>"""
    blocks = parse_filing_html(html)
    assert all(b.kind == "text" for b in blocks)
    assert any("SECURITIES" in b.text for b in blocks)


def test_section_header_detection() -> None:
    assert match_section_header("Item 1A. Risk Factors") == (None, "Item 1A. Risk Factors")
    assert match_section_header("PART II") == ("Part II", None)
    # Mid-sentence mentions must not switch sections.
    long = "As discussed in Item 1A above, fuel costs are volatile. " * 5
    assert match_section_header(long) == (None, None)


def test_filing_chunks_are_section_scoped_and_table_atomic() -> None:
    chunks = chunk_filing(_filing(FILING_HTML))
    sections = {c.metadata["section"] for c in chunks}
    assert any("Item 1A" in s for s in sections)
    assert any("Item 7" in s for s in sections)

    risk = next(c for c in chunks if "Item 1A" in c.metadata["section"])
    assert "Fuel price volatility" in risk.text
    assert risk.metadata["company_name"] == "Delta Air Lines"
    assert risk.metadata["doc_type"] == "10-K"
    # Contextual header prepended for embedding quality + citations.
    assert risk.text.startswith("[Delta Air Lines | 10-K | 2024-02-01 |")

    table_chunks = [c for c in chunks if c.metadata["content_kind"] == "table"]
    assert len(table_chunks) == 1  # the table is its own atomic chunk
    assert "Net income | 3,457" in table_chunks[0].text
    assert "Item 7" in table_chunks[0].metadata["section"]


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

TRANSCRIPT_HTML = """
<p>Morgan Stanley Laguna Conference</p>
<p>CORRECTED TRANSCRIPT: Delta Air Lines, Inc.(DAL-US), 12-September-2024</p>
<h6>Corporate Participants</h6>
<ul><li class='small'>Ravi Shanker, Analyst</li></ul>
<div class="page-header"><h6 class="text-center">MANAGEMENT DISCUSSION SECTION</h6></div>
<div class="speech-odd"><div class="speaker">Ravi Shanker, Analyst</div>
<p>Welcome to day two of the conference.</p></div>
<div class="page-header"><h6 class="text-center">Q&amp;A</h6></div>
<div class="speech-odd"><div class="speaker">Ravi Shanker, Analyst - Question</div>
<p>How is demand trending this quarter?</p></div>
<div class="speech-even"><div class="speaker">Glen Hauenstein, President - Answer</div>
<p>Demand is solid and unit revenues inflected positive in September.</p></div>
"""


def test_transcript_turns_speakers_and_sections() -> None:
    turns = parse_transcript_html(TRANSCRIPT_HTML)
    qa = [t for t in turns if t.section == "Q&A"]
    assert [t.qa_role for t in qa] == ["Question", "Answer"]
    answer = qa[1]
    assert answer.speaker == "Glen Hauenstein"
    assert answer.speaker_title == "President"
    assert "unit revenues" in answer.text


def test_transcript_chunks_keep_attribution_and_pack_qa_together() -> None:
    chunks = chunk_transcript(_transcript(TRANSCRIPT_HTML))
    qa_chunks = [c for c in chunks if c.metadata["section"] == "Q&A"]
    assert len(qa_chunks) == 1  # short Q + A pack into one chunk
    chunk = qa_chunks[0]
    assert "Ravi Shanker (Analyst) — Question:" in chunk.text
    assert "Glen Hauenstein (President) — Answer:" in chunk.text
    assert chunk.metadata["speakers"] == ["Ravi Shanker", "Glen Hauenstein"]
    # Mixed Q+A chunk carries no single qa_role.
    assert "qa_role" not in chunk.metadata
    # A chunk never spans sections.
    assert all(
        c.metadata["section"] in {"Preamble", "Corporate Participants",
                                  "Management Discussion", "Q&A"}
        for c in chunks
    )


def test_transcript_title_from_preamble() -> None:
    doc = _transcript(TRANSCRIPT_HTML)
    chunk_transcript(doc)
    assert doc.title == "Morgan Stanley Laguna Conference"
