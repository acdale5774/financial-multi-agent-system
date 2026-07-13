"""Chunk and index SEC filings + earnings/conference transcripts into pgvector.

Pipeline (per document):
    load -> parse HTML (html_text) -> chunk (this module) -> embed -> upsert

Chunking strategy (designed for financial/legal text — see be/README.md for
the full rationale):
- **Section-aware**: 10-K/10-Q "Item" headers (Item 1A Risk Factors, Item 7
  MD&A, ...) are hard boundaries — a chunk never spans two Items, and every
  chunk carries its section in metadata for filtered retrieval.
- **Table-atomic**: financial tables are linearized row-by-row and kept whole
  (split only by row-groups when enormous); splitting a table mid-row destroys
  the label↔value alignment that makes it retrievable.
- **Turn-aware (transcripts)**: the unit is the speaker turn; question/answer
  attribution survives chunking, and Management Discussion vs Q&A is tracked.
- **Contextual header**: each chunk is prefixed with
  "[Company | doc type | date | section]" so the embedding carries document
  context even when the raw passage is generic boilerplate.

Sizing: ~2,000 chars (~500 tokens) target, 3,000 max. Boundaries are semantic
(paragraph/turn/section) rather than fixed windows; only oversized single
units are hard-split, with a 200-char overlap to preserve continuity.

Metadata schema (JSONB on every chunk — enables precise filtered retrieval):
    company_name, octus_company_id, sub_industry,
    source_type ('SEC Filing' | 'Transcript'), doc_type ('10-K'|'10-Q'|...),
    document_date (ISO), title, section, content_kind ('text'|'table'),
    speakers (transcripts), qa_role (transcripts)

Run it:
    python -m ingestion.document_ingest --init-db            # everything
    python -m ingestion.document_ingest --limit 5            # smoke test
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ingestion.html_text import (
    Block,
    Turn,
    match_section_header,
    parse_filing_html,
    parse_transcript_html,
)

# Chunk sizing in characters (≈4 chars/token → ~500-token target).
TARGET_CHARS = 2_000
MAX_CHARS = 3_000
SPLIT_OVERLAP_CHARS = 200


@dataclass
class DocumentChunk:
    """A single retrievable slice of a source document."""

    document_id: str  # source-system (Octus) document id
    chunk_index: int
    text: str
    metadata: dict[str, Any]


@dataclass
class SourceDocument:
    """A parsed source document ready for chunking."""

    external_id: str
    source_type: str  # 'SEC Filing' | 'Transcript'
    doc_type: str  # '10-K' | '10-Q' | 'Transcript'
    document_date: date | None
    company: dict[str, str | None]  # octus_company_id, company_name, sub_industry
    title: str | None
    html: str


# ---------------------------------------------------------------------------
# Loading the case-study corpus
# ---------------------------------------------------------------------------


def _company_index(data_dir: Path) -> dict[str, dict[str, str | None]]:
    """Map every Octus entity id -> its parent company record."""
    records = json.loads((data_dir / "company_metadata.json").read_text())
    index: dict[str, dict[str, str | None]] = {}
    for rec in records:
        company = {
            "octus_company_id": rec.get("octus_company_id"),
            "company_name": rec.get("company_name"),
            "sub_industry": rec.get("sub_industry"),
        }
        for entity_id in str(rec.get("company_ids", "")).split(","):
            entity_id = entity_id.strip()
            if entity_id:
                index[entity_id] = company
        if rec.get("octus_company_id"):
            index[str(rec["octus_company_id"])] = company
    return index


_UNKNOWN_COMPANY: dict[str, str | None] = {
    "octus_company_id": None,
    "company_name": None,
    "sub_industry": None,
}


def _parse_doc_date(raw: str | None) -> date | None:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%Y%m%d %H%M%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def load_documents(data_dir: Path, *, limit: int | None = None) -> Iterable[SourceDocument]:
    """Yield all SEC filings then all transcripts from the case-study folder.

    `limit` caps each source type independently (smoke-testing aid).
    """
    companies = _company_index(data_dir)

    filings_meta = json.loads((data_dir / "sec_filings_metadata.json").read_text())
    for meta in filings_meta[: limit or len(filings_meta)]:
        doc_id = meta["document_id"]
        html_path = data_dir / "sec_html" / f"{doc_id}.html"
        if not html_path.exists():
            print(f"warning: missing HTML for filing {doc_id}; skipped")
            continue
        entity_ids = meta.get("company_id") or []
        company = next(
            (companies[str(e)] for e in entity_ids if str(e) in companies),
            _UNKNOWN_COMPANY,
        )
        yield SourceDocument(
            external_id=doc_id,
            source_type=meta.get("source_type", "SEC Filing"),
            doc_type=meta.get("document_type", "SEC Filing"),
            document_date=_parse_doc_date(meta.get("document_date")),
            company=company,
            title=None,  # filled from the parsed cover page below
            html=html_path.read_text(errors="replace"),
        )

    transcripts = json.loads((data_dir / "transcripts.json").read_text())
    for meta in transcripts[: limit or len(transcripts)]:
        company = companies.get(str(meta.get("company_id")), _UNKNOWN_COMPANY)
        yield SourceDocument(
            external_id=meta["document_id"],
            source_type=meta.get("source_type", "Transcript"),
            doc_type=meta.get("document_type", "Transcript"),
            document_date=_parse_doc_date(meta.get("document_date")),
            company=company,
            title=None,
            html=meta.get("body", ""),
        )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _context_header(doc: SourceDocument, section: str) -> str:
    parts = [
        doc.company.get("company_name") or "Unknown company",
        doc.doc_type,
        doc.document_date.isoformat() if doc.document_date else "undated",
        section,
    ]
    return "[" + " | ".join(parts) + "]\n"


def _base_metadata(doc: SourceDocument, section: str, kind: str) -> dict[str, Any]:
    return {
        "company_name": doc.company.get("company_name"),
        "octus_company_id": doc.company.get("octus_company_id"),
        "sub_industry": doc.company.get("sub_industry"),
        "source_type": doc.source_type,
        "doc_type": doc.doc_type,
        "document_date": doc.document_date.isoformat() if doc.document_date else None,
        "title": doc.title,
        "section": section,
        "content_kind": kind,
    }


def _hard_split(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """Split an oversized unit at line/space boundaries with overlap."""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Prefer a newline, then a space, near the end of the window.
            cut = text.rfind("\n", start + max_chars // 2, end)
            if cut == -1:
                cut = text.rfind(" ", start + max_chars // 2, end)
            if cut != -1:
                end = cut
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


@dataclass
class _Assembler:
    """Packs (text, kind) units into chunks, flushing on section change."""

    doc: SourceDocument
    chunks: list[DocumentChunk] = field(default_factory=list)
    _buf: list[str] = field(default_factory=list)
    _buf_len: int = 0
    _section: str = ""
    _kind: str = "text"
    _extra: dict[str, Any] = field(default_factory=dict)

    def add(self, text: str, *, section: str, kind: str = "text", **extra: Any) -> None:
        if section != self._section or extra != self._extra or (kind == "table") or (
            self._kind == "table"
        ):
            self.flush()
            self._section, self._kind, self._extra = section, kind, extra
        if self._buf_len + len(text) > MAX_CHARS and self._buf:
            self.flush()
            self._section, self._kind, self._extra = section, kind, extra
        self._buf.append(text)
        self._buf_len += len(text) + 1
        if self._buf_len >= TARGET_CHARS:
            self.flush()
            self._section, self._kind, self._extra = section, kind, extra

    def flush(self) -> None:
        if not self._buf:
            return
        body = "\n".join(self._buf).strip()
        self._buf, self._buf_len = [], 0
        if not body:
            return
        header = _context_header(self.doc, self._section)
        pieces = (
            _hard_split(body, max_chars=MAX_CHARS, overlap=SPLIT_OVERLAP_CHARS)
            if len(body) > MAX_CHARS
            else [body]
        )
        for piece in pieces:
            metadata = _base_metadata(self.doc, self._section, self._kind)
            metadata.update(self._extra)
            self.chunks.append(
                DocumentChunk(
                    document_id=self.doc.external_id,
                    chunk_index=len(self.chunks),
                    text=header + piece,
                    metadata=metadata,
                )
            )


def chunk_filing(doc: SourceDocument) -> list[DocumentChunk]:
    """Section-aware, table-atomic chunking of a 10-K/10-Q filing."""
    blocks: list[Block] = parse_filing_html(doc.html)
    if doc.title is None:
        doc.title = _filing_title(blocks, doc)

    assembler = _Assembler(doc)
    part: str | None = None
    item: str | None = None
    for block in blocks:
        if block.kind == "text":
            new_part, new_item = match_section_header(block.text)
            if new_part:
                part, item = new_part, None
                continue
            if new_item:
                item = new_item
                continue
        section = " · ".join(x for x in (part, item) if x) or "Front matter"
        assembler.add(block.text, section=section, kind=block.kind)
    assembler.flush()
    return assembler.chunks


def _filing_title(blocks: list[Block], doc: SourceDocument) -> str:
    name = doc.company.get("company_name") or "Unknown company"
    return f"{name} {doc.doc_type}" + (
        f" ({doc.document_date.isoformat()})" if doc.document_date else ""
    )


def chunk_transcript(doc: SourceDocument) -> list[DocumentChunk]:
    """Turn-aware chunking of an earnings/conference call transcript.

    Consecutive turns within one section are packed together (a question and
    its answer usually land in the same chunk); a chunk never spans sections.
    Speaker attribution is kept inline and in metadata (`speakers`, and
    `qa_role` when the whole chunk is one side of the exchange).
    """
    turns: list[Turn] = parse_transcript_html(doc.html)
    if doc.title is None:
        preamble = [t for t in turns if t.section == "Preamble"]
        doc.title = preamble[0].paragraphs[0][:200] if preamble else "Transcript"

    chunks: list[DocumentChunk] = []
    buf: list[str] = []
    buf_len = 0
    section = ""
    speakers: list[str] = []
    qa_roles: set[str | None] = set()

    def flush() -> None:
        nonlocal buf, buf_len, speakers, qa_roles
        body = "\n\n".join(buf).strip()
        buf, buf_len = [], 0
        chunk_speakers, chunk_roles = speakers, qa_roles
        speakers, qa_roles = [], set()
        if not body:
            return
        metadata = _base_metadata(doc, section, "text")
        if chunk_speakers:
            metadata["speakers"] = chunk_speakers
        if len(chunk_roles) == 1 and None not in chunk_roles:
            metadata["qa_role"] = next(iter(chunk_roles))
        chunks.append(
            DocumentChunk(
                document_id=doc.external_id,
                chunk_index=len(chunks),
                text=_context_header(doc, section) + body,
                metadata=metadata,
            )
        )

    for turn in turns:
        text = turn.text
        if not text:
            continue
        attribution = None
        if turn.speaker:
            attribution = turn.speaker + (
                f" ({turn.speaker_title})" if turn.speaker_title else ""
            )
            if turn.qa_role:
                attribution += f" — {turn.qa_role}"
            text = f"{attribution}:\n{text}"

        if turn.section != section and buf:
            flush()
        section = turn.section

        # Oversized single turn (a long prepared-remarks monologue): hard-split.
        pieces = (
            _hard_split(text, max_chars=MAX_CHARS, overlap=SPLIT_OVERLAP_CHARS)
            if len(text) > MAX_CHARS
            else [text]
        )
        for i, piece in enumerate(pieces):
            if i > 0 and attribution:
                piece = f"{attribution} (cont.):\n{piece}"
            if buf and buf_len + len(piece) > MAX_CHARS:
                flush()
                section = turn.section
            buf.append(piece)
            buf_len += len(piece) + 2
            if turn.speaker and turn.speaker not in speakers:
                speakers.append(turn.speaker)
            qa_roles.add(turn.qa_role)
            if buf_len >= TARGET_CHARS:
                flush()
                section = turn.section
    flush()
    return chunks


def chunk_document(doc: SourceDocument) -> list[DocumentChunk]:
    """Chunk any supported document type."""
    if doc.source_type == "Transcript":
        return chunk_transcript(doc)
    return chunk_filing(doc)


# ---------------------------------------------------------------------------
# Embed + index
# ---------------------------------------------------------------------------


def embed_and_index_chunks(
    doc: SourceDocument,
    chunks: list[DocumentChunk],
    *,
    database_url: str | None = None,
) -> int:
    """Embed a document's chunks and upsert document + chunks into Postgres.

    Idempotent: the document row is upserted on `external_id` and its chunks
    are replaced wholesale, so re-ingesting refreshes rather than duplicates.
    """
    if not chunks:
        return 0

    from pgvector.psycopg import register_vector
    from psycopg.types.json import Jsonb

    from db.connection import get_connection
    from ingestion.embeddings import get_embedder

    embedder = get_embedder()
    vectors = embedder.embed([c.text for c in chunks])

    with get_connection(database_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents
                    (external_id, source, doc_type, title,
                     octus_company_id, company_name, sub_industry, document_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (external_id) DO UPDATE
                    SET doc_type = EXCLUDED.doc_type,
                        title = EXCLUDED.title,
                        company_name = EXCLUDED.company_name,
                        document_date = EXCLUDED.document_date
                RETURNING id
                """,
                (
                    doc.external_id,
                    doc.source_type,
                    doc.doc_type,
                    doc.title,
                    doc.company.get("octus_company_id"),
                    doc.company.get("company_name"),
                    doc.company.get("sub_industry"),
                    doc.document_date,
                ),
            )
            db_doc_id = cur.fetchone()["id"]

            cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (db_doc_id,))
            cur.executemany(
                """
                INSERT INTO document_chunks
                    (document_id, chunk_index, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (db_doc_id, c.chunk_index, c.text, Jsonb(c.metadata), vectors[i])
                    for i, c in enumerate(chunks)
                ],
            )
        conn.commit()
    return len(chunks)


# ---------------------------------------------------------------------------
# Orchestration + CLI
# ---------------------------------------------------------------------------


def ingest(
    data_dir: str | Path | None = None,
    *,
    limit: int | None = None,
    init_db: bool = False,
    database_url: str | None = None,
) -> tuple[int, int]:
    """Chunk, embed, and index the whole corpus. Returns (documents, chunks)."""
    from core.config import settings

    directory = Path(data_dir or settings.documents_data_dir)
    if init_db:
        from db.connection import apply_schema

        apply_schema(database_url)

    n_docs = n_chunks = 0
    for doc in load_documents(directory, limit=limit):
        chunks = chunk_document(doc)
        stored = embed_and_index_chunks(doc, chunks, database_url=database_url)
        n_docs += 1
        n_chunks += stored
        print(f"[{n_docs:>3}] {doc.doc_type:<10} {doc.company.get('company_name') or '?':<40}"
              f" {stored:>4} chunks  ({doc.external_id})")
    return n_docs, n_chunks


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Chunk and index SEC filings + transcripts into pgvector."
    )
    parser.add_argument("--data-dir", default=None, help="Path to case_study_data.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max documents per source type (smoke testing).")
    parser.add_argument("--init-db", action="store_true",
                        help="Apply db/schema.sql before indexing.")
    args = parser.parse_args(argv)

    docs, chunks = ingest(args.data_dir, limit=args.limit, init_db=args.init_db)
    print(f"Done. {docs} documents -> {chunks} chunks indexed.")


if __name__ == "__main__":
    main()
