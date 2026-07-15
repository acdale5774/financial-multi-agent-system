# Retrieval

The retrieval layer sits between stored data and the agent. It answers the
question *"what information is relevant to this query?"* from two complementary
sources:

| Source | Store | Access pattern | Used for |
| --- | --- | --- | --- |
| SimFin financials | Postgres tables | Read-only SQL | Precise numbers, ratios, time series |
| SEC filings / transcripts | Postgres (pgvector + full-text) | Dense / lexical / hybrid search | Narrative context, qualitative statements |

## How it's used

The [tools](../tools) wrap these access patterns so the [agent](../agents) can
call them:

- `tools/sql_tool.py` — structured retrieval over financials.
- `tools/document_search_tool.py` — dense/lexical/hybrid retrieval over
  document chunks. Every result carries document id, title, section, date,
  speaker attribution, and its retrieval trace (strategy, per-leg ranks,
  fused score) for evaluation/debugging.

(Terminology note: the *agent route* named "hybrid" in `agents/` — the
specialist holding both toolsets — is unrelated to hybrid *retrieval* here.)

## Document retrieval strategies

`search_documents` supports three strategies, selected by
`RETRIEVAL_STRATEGY` (see `core/config.py`; per-call override via the
`strategy` argument):

| Strategy | Mechanism | Strong at | Weak at |
| --- | --- | --- | --- |
| `dense` | Cosine over 512-dim model2vec embeddings, pgvector HNSW index | Paraphrase — queries sharing no tokens with the evidence | Rare exact terms (TRASM, ASC 606) the static embedder averages away |
| `lexical` | Postgres full-text (`content_tsv`, GIN), `websearch_to_tsquery` then a quorum-ranked OR fallback | Exact jargon, acronyms, standard/product identifiers | True paraphrase; term-frequency ranking has **no IDF** (see limitations) |
| `hybrid` | Both legs at `RETRIEVAL_LEG_K` depth, fused with Reciprocal Rank Fusion | Robustness across both query families | Can dilute a single leg's lone top hit (consensus effect) |

Every strategy applies the **same metadata filters** (company, doc type,
section, content kind, qa_role, date window) inside its query — filters are
compiled once (`_compile_filters`) and shared by both legs, so dense and
lexical can never drift apart on scoping.

### Why dense and lexical are complementary

Dense retrieval maps meaning: "aircraft sitting on the ground waiting for
engine inspections" finds Pratt & Whitney groundings passages that share no
vocabulary with the query. Lexical retrieval maps *tokens*: a query naming
`CASM ex-fuel` or `ASC 606` must surface the chunks containing exactly those
strings — the place embeddings (especially static ones, but transformer
embeddings too) are thinnest, because rare identifiers have poorly-learned
representations. Neither subsumes the other; the benchmark's per-category
table shows each winning cases the other misses.

### Why metadata filtering does not replace lexical retrieval

Filters solve the *entity precision* half of the exact-match problem
(company, form type, section, speaker role are indexed structured values —
stronger than keyword matching). They cannot rank by terms *inside* chunk
prose: "which chunks discuss TRASM" is not expressible as a filter. Lexical
retrieval covers that residual — which is why both exist.

### How RRF works

Each leg returns an independent ranking. A chunk's fused score is

```
score = Σ over legs   1 / (rrf_k + rank_in_leg)     # missing from a leg ⇒ no term
```

with `rrf_k = 60` (the conventional constant; larger flattens rank weight).
Results are deduplicated by `(document_id, chunk_index)` and ordered by fused
score with a deterministic id tie-break. **Raw scores are never mixed**:
cosine similarity and `ts_rank_cd` are not on comparable scales; RRF uses
only ranks, which is the entire point of choosing it.

### Why PostgreSQL full-text search (and not another service)

The same reasoning as pgvector-over-Pinecone: the corpus is ~31k chunks in a
database we already run. A `tsvector` generated column + GIN index gives
lexical retrieval with the same metadata filters in the same query, zero new
infrastructure, and one backup/deploy target. Elasticsearch et al. earn their
operational cost at orders of magnitude more documents or when BM25-grade
ranking is demonstrably needed — see the trigger conditions below.

**What is indexed**: `content` only. Chunk content already begins with the
contextual header `[Company | doc type | date | section]` and transcript
chunks keep speaker attribution inline, so body + header + section + speakers
are covered by one column with no second source of truth.

**Tokenization** (verified on this corpus, PG16, `english` config):
hyphenated jargon indexes as compound and parts (`CASM-ex` →
`casm-ex, casm, ex`), so websearch phrase queries match it; acronyms (TRASM,
DOCSIS) and standard ids (ASC 606) tokenize cleanly. No custom
parser/dictionary — the benchmark hasn't demonstrated the need.

**Query construction**: strict `websearch_to_tsquery` first (safe on
arbitrary text, preserves phrases). It ANDs all non-stopwords, so long
natural-language questions often match nothing; the leg then falls back to
OR-ing the query lexemes ranked by **quorum** (distinct terms matched) before
`ts_rank_cd`. Quorum exists because Postgres ranking has no IDF: without it,
a chunk repeating a common term ("Delta" ×30) outranks every chunk containing
the rare term that carries the query's meaning (TRASM) — a failure the
benchmark caught (case `ex-trasm-delta`), fixed with stock operators only.

## The benchmark (how the default was chosen)

A 29-case labeled benchmark (`evals/retrieval_cases.py`; runner
`python -m evals.retrieval`; committed results in
[`../evals/retrieval_report.md`](../evals/retrieval_report.md)) compares the
three strategies on identical cases. Labels are anchor passages located by
reading the source documents and verified with SQL substring scans — never
by running the retriever, so ground truth is not circular. 2026-07 results
(26 supported cases, k=10):

| Strategy | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | --- |
| dense | 58% | 65% | 0.481 |
| lexical | **77%** | **81%** | 0.653 |
| hybrid | 69% | 77% | **0.655** |

**Default: `hybrid`.** The reasoning, stated honestly:

- vs **dense**: hybrid is decisively better (+12 pts Recall@10, +0.17 MRR).
  Dense-only is the weakest configuration with the current static embedder.
- vs **lexical**: the gap (1–2 cases) is inside this benchmark's noise floor,
  and the labels have a known lexical-friendly bias — anchor-substring ground
  truth counts a dense hit on a synonymous passage as a miss. Hybrid hedges
  that bias, keeps dense's paraphrase ability for query shapes the benchmark
  under-samples, and automatically benefits when the embedder is upgraded.
- **Lexical is a legitimate alternative** if per-query latency/cost matters
  most: it skips the embedding lookup entirely and posted the best measured
  recall. A definitive lexical-vs-hybrid ruling needs a larger benchmark.

Hybrid retrieval is **not universally superior** — it lost individual cases
to each single leg (see the per-case table in the report), and on a corpus
with a strong transformer embedder the trade-offs could look different.
It is the best-hedged default *for this corpus and this embedder, measured*.

## Known limitations

- **No IDF in lexical ranking.** Quorum mitigates but does not fix rare-term
  starvation: `ex-trasm-delta` still misses at k=10 (first relevant at rank
  11) because "delta + trend" quorum-ties with "delta + TRASM".
- **No abstention.** All strategies return k results even for queries with no
  relevant evidence (the three `unsupported` cases): similarity always
  produces neighbours, and the lexical OR-fallback always matches something.
  Downstream, the agent's judgment + citation validation are the guard.
- **Benchmark scale.** 29 cases ranks strategies directionally; treat
  few-point differences as noise (stated in the report itself).
- **Duplicate documents.** The corpus stores some filings 2–4× under distinct
  external ids (242 physical / 194 logical documents), so top-k can be padded
  with near-identical chunks. An ingestion-side dedup would raise effective
  diversity at every k — out of scope for the retrieval layer.

## Upgrade triggers (measure first, then buy)

- **Re-run the benchmark** after any chunker, embedder, or corpus change:
  `python -m evals.retrieval --write-report` (needs only Postgres).
- **Upgrade the embedder** (e.g. Voyage `voyage-finance-2` via the `Embedder`
  protocol) when dense/semantic recall is the binding constraint — the
  static model2vec embedder is a local-dev hardware constraint, not a
  production recommendation; expect dense and hybrid numbers to rise.
- **Add a reranker or BM25-grade engine** (e.g. a cross-encoder over the
  fused top-50, or a BM25 extension) when rare-term cases like
  `ex-trasm-delta` matter in production traffic — that failure is precisely
  an IDF/reranking gap, now quantified by the benchmark.
- **Grow the benchmark** (more paraphrase cases, labeled by a second person)
  before trusting any single-digit-point conclusion.
