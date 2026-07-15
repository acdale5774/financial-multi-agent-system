"""The labeled document-retrieval benchmark: queries + ground-truth relevance.

Each case defines its relevant chunks *declaratively* — document constraints
(company, doc type, section, date window) plus one or more **anchor phrases**
that must appear in the chunk text. The harness resolves that definition
against the chunk store with plain SQL (`ILIKE` substring scans); the retriever
under test is never consulted while building ground truth, so labels cannot be
circular.

How the labels were produced (methodology):
  1. Anchor phrases were chosen by reading the source documents/corpus for
     facts that answer the query (e.g. Delta's 10-K names CrowdStrike; JetBlue's
     10-K attributes engine groundings to "powdered metal").
  2. Every anchor was verified present in the chunk store with a direct
     substring scan (never a similarity search), and the match count at
     authoring time is recorded in `notes` — the same "measured, not invented"
     rule as `evals/cases.py`.
  3. The chunker slices documents verbatim, so "chunks containing this source
     passage" identifies the same evidence regardless of chunk boundaries —
     the labels survive re-chunking, unlike hard-coded chunk ids.

Relevance rule (deterministic; implemented in `evals/retrieval.py`):
  A chunk is relevant iff ALL set document-level constraints hold
  (company ∈ relevant_companies, doc_type, section ILIKE pattern, date window,
  content_kind, speaker substring) AND its content contains at least one
  anchor phrase (case-insensitive substring). `unsupported` cases have no
  anchors — their relevant set is empty by definition, and correct behaviour
  is returning nothing that pretends to be evidence.

Known limitations (state them, don't hide them):
  - 29 cases is a smoke-test scale, like the six text-to-SQL cases: enough to
    compare strategies directionally, far too few to certify recall in
    production. Treat deltas of a few percent as noise.
  - Anchor-substring relevance is high-precision but not exhaustive: a chunk
    can answer a query without containing the anchor (synonymous passages are
    counted as misses). Recall numbers are therefore lower bounds.
  - The corpus contains near-duplicate filings (some documents appear twice
    with different external ids); relevant sets include all copies, so
    duplication does not bias any one strategy.
  - Category sizes (3-6 cases) are too small for per-category significance;
    per-category tables are for locating failures, not ranking strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CATEGORIES = (
    "semantic",
    "exact_term",
    "section_scoped",
    "speaker_scoped",
    "metadata_filtered",
    "cross_document",
    "unsupported",
)


@dataclass(frozen=True)
class RetrievalCase:
    """One labeled retrieval case.

    Attributes:
        id: Stable identifier.
        category: One of CATEGORIES (also the scorecard bucket).
        query: Natural-language search query fed to `search_documents`.
        why: Why the labeled evidence answers the query (reviewer context).
        filters: Search-time metadata filters passed to `search_documents`
            (only set where the *use case* is a filtered search:
            metadata_filtered / speaker_scoped). Ground truth never depends
            on these — it depends on the relevance fields below.
        anchors: Chunk-content substrings (case-insensitive) that mark a chunk
            as relevant. Empty ⇔ category == 'unsupported'.
        relevant_companies: documents.company_name values the evidence may
            come from. Empty = any company.
        relevant_doc_type: Constrain evidence to one doc type (e.g. '10-K').
        relevant_section_ilike: ILIKE pattern the chunk's section must match.
        relevant_date_from / relevant_date_to: Evidence document-date window.
        relevant_content_kind: Constrain to 'text' or 'table' chunks.
        relevant_speaker: Substring that must appear in the chunk's transcript
            `speakers` metadata (speaker-attribution ground truth).
        relevant_qa_role: Constrain to 'Answer' or 'Question' transcript
            chunks — "what management SAID" excludes analyst questions.
        notes: Authoring-time measurements and context.
    """

    id: str
    category: str
    query: str
    why: str
    filters: dict[str, Any] = field(default_factory=dict)
    anchors: tuple[str, ...] = ()
    relevant_companies: tuple[str, ...] = ()
    relevant_doc_type: str | None = None
    relevant_section_ilike: str | None = None
    relevant_date_from: str | None = None
    relevant_date_to: str | None = None
    relevant_content_kind: str | None = None
    relevant_speaker: str | None = None
    relevant_qa_role: str | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# The cases. Anchor match counts measured 2026-07-14 against the ingested
# corpus (242 documents, 30,939 chunks).
# ---------------------------------------------------------------------------

RETRIEVAL_CASES: list[RetrievalCase] = [
    # -- semantic: paraphrased questions; the query avoids the anchor tokens --
    RetrievalCase(
        id="sem-delta-it-outage",
        category="semantic",
        query=(
            "How was Delta Air Lines affected by the worldwide technology "
            "outage in the summer of 2024?"
        ),
        why=(
            "Delta's filings and calls discuss the July 2024 outage by vendor "
            "name (CrowdStrike); the query paraphrases the event without it."
        ),
        anchors=("CrowdStrike",),
        relevant_companies=("Delta Air Lines",),
        notes="38 matching chunks measured (10-K risk factors, 10-Qs, transcripts).",
    ),
    RetrievalCase(
        id="sem-jetblue-engine-groundings",
        category="semantic",
        query=(
            "Why does JetBlue have aircraft sitting on the ground waiting "
            "for engine inspections?"
        ),
        why=(
            "JetBlue attributes groundings to contaminated powdered metal in "
            "Pratt & Whitney engines; the query names neither term."
        ),
        anchors=("powdered metal",),
        relevant_companies=("JetBlue",),
        notes="17 matching chunks measured.",
    ),
    RetrievalCase(
        id="sem-rivian-r2-affordability",
        category="semantic",
        query="How is Rivian bringing down the cost of its smaller upcoming vehicle?",
        why=(
            "Rivian's calls discuss reducing the R2 bill of materials to hit "
            "its target price points; the query paraphrases without 'R2' or "
            "'bill of materials'."
        ),
        anchors=("bill of materials",),
        relevant_companies=("Rivian Automotive Inc.",),
        notes="34 matching chunks measured.",
    ),
    RetrievalCase(
        id="sem-sixflags-combination",
        category="semantic",
        query=(
            "What are the terms of Six Flags' merger of equals with another "
            "amusement park operator?"
        ),
        why="The counterparty (Cedar Fair) is named only in the filings.",
        anchors=("Cedar Fair",),
        relevant_companies=("Six Flags Entertainment Corporation",),
        notes="37 matching chunks measured.",
    ),
    RetrievalCase(
        id="sem-cableone-fwa-competition",
        category="semantic",
        query=(
            "What competition does Cable One face from wireless home "
            "internet offerings?"
        ),
        why=(
            "Cable One's disclosures call this 'fixed wireless' competition; "
            "the query uses the consumer phrasing instead."
        ),
        anchors=("fixed wireless",),
        relevant_companies=("Cable One, Inc.",),
        notes="24 matching chunks measured.",
    ),
    RetrievalCase(
        id="sem-vistance-identity",
        category="semantic",
        query="What kind of company is Vistance Networks and what does it sell?",
        why=(
            "The corpus's Vistance documents are CommScope filings under a "
            "fictional name: the contextual header says Vistance, the body "
            "says CommScope. Tests retrieval when header and body entity "
            "names disagree."
        ),
        anchors=("CommScope",),
        relevant_companies=("Vistance Networks Inc.",),
        relevant_section_ilike="%item 1. business%",
        notes="43 matching chunks measured in Item 1 Business sections.",
    ),
    # -- exact_term: the query contains the exact jargon/acronym/standard ----
    RetrievalCase(
        id="ex-casm-ex-jetblue",
        category="exact_term",
        query="What did JetBlue report for CASM ex-fuel?",
        why=(
            "CASM ex-fuel is a unit-cost metric named exactly in filings; "
            "static embeddings are weakest on rare hyphenated jargon."
        ),
        anchors=("CASM ex",),
        relevant_companies=("JetBlue",),
        notes=(
            "63 matching chunks measured. Anchor 'CASM ex' covers both "
            "'CASM ex-fuel' and 'CASM ex fuel' spellings."
        ),
    ),
    RetrievalCase(
        id="ex-trasm-delta",
        category="exact_term",
        query="How did Delta's TRASM trend?",
        why="TRASM (total revenue per available seat mile) is an exact acronym.",
        anchors=("TRASM",),
        relevant_companies=("Delta Air Lines",),
        notes="82 matching chunks measured.",
    ),
    RetrievalCase(
        id="ex-asc606",
        category="exact_term",
        query="Which revenue recognition disclosures reference ASC 606?",
        why=(
            "ASC 606 is an accounting-standard identifier; any company's "
            "disclosure citing it is relevant (3 companies do)."
        ),
        anchors=("ASC 606",),
        notes="51 matching chunks measured across 3 companies.",
    ),
    RetrievalCase(
        id="ex-docsis4",
        category="exact_term",
        query="What are the plans for DOCSIS 4.0 network upgrades?",
        why="DOCSIS 4.0 is an exact technology-standard identifier.",
        anchors=("DOCSIS 4.0",),
        notes="74 matching chunks measured (cable operators + Vistance).",
    ),
    RetrievalCase(
        id="ex-viasat3",
        category="exact_term",
        query="What is the status of the ViaSat-3 satellite program?",
        why="ViaSat-3 is a hyphenated product identifier unique to Viasat.",
        anchors=("ViaSat-3",),
        relevant_companies=("Viasat Inc.",),
        notes="138 matching chunks measured.",
    ),
    RetrievalCase(
        id="ex-aadvantage",
        category="exact_term",
        query=(
            "What did American Airlines disclose about the AAdvantage "
            "loyalty program?"
        ),
        why="AAdvantage is an exact brand name unique to American.",
        anchors=("AAdvantage",),
        relevant_companies=("American Airlines Group",),
        notes="192 matching chunks measured.",
    ),
    # -- section_scoped: evidence must come from a specific filing section ---
    RetrievalCase(
        id="sec-delta-fuel-risk",
        category="section_scoped",
        query="What risk factors does Delta disclose about fuel prices?",
        why=(
            "The question targets Item 1A specifically — fuel is discussed "
            "all over the filings, but only Risk Factors passages answer it."
        ),
        anchors=("fuel price",),
        relevant_companies=("Delta Air Lines",),
        relevant_section_ilike="%item 1a%",
        notes="5 matching chunks measured — a tight target.",
    ),
    RetrievalCase(
        id="sec-american-labor-risk",
        category="section_scoped",
        query=(
            "What does American Airlines say about union labor relations "
            "in its risk factors?"
        ),
        why="Labor-relations risk lives in Item 1A; MD&A mentions don't answer it.",
        anchors=("union",),
        relevant_companies=("American Airlines Group",),
        relevant_section_ilike="%item 1a%",
        notes="28 matching chunks measured.",
    ),
    RetrievalCase(
        id="sec-ford-market-risk-commodity",
        category="section_scoped",
        query=(
            "How does Ford describe commodity price exposure in its "
            "quantitative and qualitative market risk disclosures?"
        ),
        why="Item 7A is the canonical home of commodity-hedging exposure.",
        anchors=("commodity",),
        relevant_companies=("Ford Motor",),
        relevant_section_ilike="%item 7a%",
        notes="15 matching chunks measured.",
    ),
    RetrievalCase(
        id="sec-charter-business-brand",
        category="section_scoped",
        query=(
            "How does Charter describe its Spectrum brand in the business "
            "overview of its annual report?"
        ),
        why="Item 1 Business is where the brand/product overview lives.",
        anchors=("Spectrum",),
        relevant_companies=("Charter Communications",),
        relevant_section_ilike="%item 1. business%",
        notes="137 matching chunks measured.",
    ),
    # -- speaker_scoped: transcript attribution is part of the ground truth --
    RetrievalCase(
        id="spk-hauenstein-premium",
        category="speaker_scoped",
        query="What has Delta's president said about premium revenue?",
        why=(
            "Only answers spoken by Glen Hauenstein (President) count — an "
            "analyst asking about premium is not evidence of what management said."
        ),
        filters={"company_name": "Delta Air Lines", "qa_role": "Answer"},
        anchors=("premium",),
        relevant_companies=("Delta Air Lines",),
        relevant_speaker="Hauenstein",
        relevant_qa_role="Answer",
        notes="2 matching chunks measured with qa_role=Answer — the hardest target.",
    ),
    RetrievalCase(
        id="spk-scaringe-r2",
        category="speaker_scoped",
        query="What has Rivian's CEO said about the R2 program?",
        why="Attribution matters: RJ Scaringe's own statements about R2.",
        filters={"company_name": "Rivian Automotive Inc."},
        anchors=("R2",),
        relevant_companies=("Rivian Automotive Inc.",),
        relevant_speaker="Scaringe",
        notes="172 matching chunks measured (R2 dominates Rivian's calls).",
    ),
    RetrievalCase(
        id="spk-mcdonough-gross-profit",
        category="speaker_scoped",
        query="What did Rivian's CFO answer about gross profit?",
        why="CFO answers (Claire McDonough) in Q&A, not prepared remarks.",
        filters={"company_name": "Rivian Automotive Inc.", "qa_role": "Answer"},
        anchors=("gross profit",),
        relevant_companies=("Rivian Automotive Inc.",),
        relevant_speaker="McDonough",
        relevant_qa_role="Answer",
        notes="3 matching chunks measured with qa_role=Answer.",
    ),
    RetrievalCase(
        id="spk-laulis-broadband",
        category="speaker_scoped",
        query="What has Cable One's CEO said about broadband strategy?",
        why="Julia Laulis's transcript remarks about broadband.",
        filters={"company_name": "Cable One, Inc."},
        anchors=("broadband",),
        relevant_companies=("Cable One, Inc.",),
        relevant_speaker="Laulis",
        notes="46 matching chunks measured.",
    ),
    # -- metadata_filtered: the use case is a filtered search ----------------
    RetrievalCase(
        id="mf-delta-oct24-fuel",
        category="metadata_filtered",
        query="Discussion of jet fuel costs",
        why=(
            "A short generic query made precise by filters: Delta's October "
            "2024 10-Q only."
        ),
        filters={
            "company_name": "Delta Air Lines",
            "doc_type": "10-Q",
            "date_from": "2024-10-01",
            "date_to": "2024-10-31",
        },
        anchors=("fuel",),
        relevant_companies=("Delta Air Lines",),
        relevant_doc_type="10-Q",
        relevant_date_from="2024-10-01",
        relevant_date_to="2024-10-31",
        notes="56 matching chunks measured inside the window.",
    ),
    RetrievalCase(
        id="mf-jetblue-revenue-tables",
        category="metadata_filtered",
        query="Table of operating revenue figures",
        why=(
            "content_kind='table' scopes retrieval to linearized financial "
            "tables — the right target for numeric lookups."
        ),
        filters={"company_name": "JetBlue", "content_kind": "table"},
        anchors=("revenue",),
        relevant_companies=("JetBlue",),
        relevant_content_kind="table",
        notes="71 matching table chunks measured.",
    ),
    RetrievalCase(
        id="mf-ciena-transcript-fcf",
        category="metadata_filtered",
        query="Management commentary on free cash flow",
        why="doc_type='Transcript' scopes to spoken commentary, not filings.",
        filters={"company_name": "Ciena Corp.", "doc_type": "Transcript"},
        anchors=("free cash flow",),
        relevant_companies=("Ciena Corp.",),
        relevant_doc_type="Transcript",
        notes="9 matching chunks measured.",
    ),
    # -- cross_document: evidence spans companies or repeats across periods --
    RetrievalCase(
        id="xd-acp-sunset",
        category="cross_document",
        query=(
            "How did the end of the federal broadband subsidy program "
            "affect cable operators?"
        ),
        why=(
            "Both Cable One and Charter disclose the Affordable Connectivity "
            "Program wind-down; the query names neither the program nor a company."
        ),
        anchors=("Affordable Connectivity Program",),
        relevant_companies=("Cable One, Inc.", "Charter Communications"),
        notes="Evidence spans two companies' filings and calls.",
    ),
    RetrievalCase(
        id="xd-gtf-engines",
        category="cross_document",
        query=(
            "Which airlines report impacts from Pratt & Whitney engine "
            "inspections?"
        ),
        why=(
            "Completeness across companies is the challenge: JetBlue and "
            "American both disclose it."
        ),
        anchors=("Pratt & Whitney",),
        relevant_companies=("JetBlue", "American Airlines Group"),
        notes="JetBlue ~89 chunks and American ~18 chunks measured.",
    ),
    RetrievalCase(
        id="xd-delta-climate-repeat",
        category="cross_document",
        query=(
            "What climate-related risks does Delta repeat across its "
            "annual reports?"
        ),
        why=(
            "Near-duplicate Risk Factors recur in each year's 10-K — "
            "retrieval should surface the disclosures across periods."
        ),
        anchors=("climate",),
        relevant_companies=("Delta Air Lines",),
        relevant_doc_type="10-K",
        relevant_section_ilike="%item 1a%",
        notes="20 matching chunks measured across multiple fiscal years.",
    ),
    # -- unsupported: no relevant evidence exists in the corpus --------------
    RetrievalCase(
        id="un-apple-iphone",
        category="unsupported",
        query="What does Apple's 10-K say about iPhone revenue?",
        why=(
            "Apple has no documents in the corpus. The token 'iPhone' does "
            "appear in 3 unrelated chunks, so lexical matching will surface "
            "plausible-looking non-evidence — the trap this case sets."
        ),
        notes="Measured: 0 Apple documents; 3 incidental 'iPhone' mentions.",
    ),
    RetrievalCase(
        id="un-tesla-cybertruck",
        category="unsupported",
        query="What production updates has Tesla given for the Cybertruck?",
        why="Neither Tesla documents nor the token 'Cybertruck' exist in the corpus.",
        notes="Measured: 0 matches for 'Cybertruck'.",
    ),
    RetrievalCase(
        id="un-drug-pricing",
        category="unsupported",
        query=(
            "What pressure are companies facing from prescription drug "
            "pricing reform?"
        ),
        why="No pharmaceutical companies or drug-pricing disclosures exist in the corpus.",
        notes="Measured: 0 matches for 'drug pricing'.",
    ),
]
