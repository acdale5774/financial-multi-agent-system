# Text-to-SQL evaluation harness

A small, grounded evaluation set for the structured-data query agent. It exists
to answer one question with evidence instead of intuition:

> Which financial semantics does the agent actually get right — and which
> should be promoted from curated prose notes (`tools/schema_tool.py`
> `_TABLE_NOTES`) into a governed, machine-readable model *first*?

> **Scope — read this first.** This is a **targeted regression suite covering
> six high-risk financial semantics**, not a general accuracy benchmark and not
> proof that text-to-SQL is "solved." Six cases is a smoke test: enough to catch
> a regression in the traps that most often produce a confidently-wrong number,
> far too few to characterise accuracy over the space of real questions
> (paraphrases, ambiguous entities, unsupported metrics, adversarial phrasing).
> The dimensions a fuller evaluation would add — and which are already seeded —
> are named explicitly [below](#what-a-fuller-evaluation-would-measure).

## Why this exists

The structured store is a long/tall `financials` table. The semantics that make
querying it correct — revenue aggregates by SUM, cash is a period-end value,
balance sheets have no `FY` row, `eps` must be derived, "Ford" is `F` not `FORD`
— currently live **procedurally**, as curated notes fed to the agent. That works,
but prose hints are not testable and don't scale to twenty more tables.

This harness makes those semantics **testable**. Each case stresses exactly one
of them and names (`semantic_rule`) the machine-readable rule a governed layer
would encode. When the agent starts failing a category — as harder questions or
new sources arrive — that category is the thing to formalize next. Until then,
the passing cases are a regression net proving the curated notes still hold.

## Two layers

| Layer | Needs | What it checks |
|---|---|---|
| **Ground truth** (`run_ground_truth`) | Postgres only | Each golden SQL, run through the same read-only SQL tool the agent uses, returns the value measured from the live DB. Deterministic. Also confirms each documented "trap" query still reproduces its wrong answer. |
| **Agent grading** (`run_agent`, `--agent`) | Postgres + `OPENAI_API_KEY` | The real NL→SQL agent answers the natural-language question; its prose is graded against the same ground truth (scale/rounding/sign-aware, reusing the citation validator's number matcher). Trap value present in the answer ⇒ failed-with-warning. |

Every value in `cases.py` was **measured from the database**, not invented.

Grading is **result equivalence, never exact SQL strings.** The agent is free to
reach the answer by any valid formulation (a self-join, a window function, a CTE);
it passes when the *value* in its prose matches the golden value under
scale/rounding/sign-aware number matching (the same matcher the citation
validator uses). This deliberately does not penalise a correct query written a
different way — which also means the suite says nothing about SQL *style*, only
about answer correctness.

## Running

```bash
# From be/ (uses the read-only SQL tool + settings.database_url)
python -m evals                      # ground truth only — no API key, deterministic
python -m evals --agent              # + grade the real agent (spends OpenAI credits)
python -m evals --category stock-aggregation   # one semantic at a time
python -m evals --agent --json       # machine-readable scorecard
```

Exit code is non-zero if **ground truth** regresses (golden SQL drifted from the
data). Agent misses are informational — they're the signal, not a build break.

Pytest coverage lives in `tests/test_evals.py`: pure tests (case set is
well-formed; the grader is scale/sign-aware) always run; the live-DB tests skip
automatically when Postgres isn't reachable.

## The cases

| Category | Stresses | Trap it guards against |
|---|---|---|
| `flow-aggregation` | Revenue sums across quarters (`SUM`) | — (SUM is correct for flows) |
| `stock-aggregation` | Cash is a period-end stock | Summing the four quarterly snapshots (231B vs 61.6B) |
| `period-mapping` | Balance sheets have no `FY`; Q4 = year-end | Filtering balance for `fiscal_period='FY'` → 0 rows |
| `derived-metric` | `eps = net_income / shares_diluted` | Expecting a stored `eps` metric |
| `entity-resolution` | "Ford" → Ford Motor (`F`) | Resolving to `FORD` = Forward Industries (176B vs 37M) |
| `computation` | Growth = (curr − prior) / prior | — |

## Current finding (2026-07, SimFin US, gpt-5.5)

Ground truth **6/6**; agent **6/6**. The agent handles all six traps with the
current curated notes — on the stock-aggregation case it returns the Q4 value
*and* explains that the Q4 snapshot is the fiscal-year-end balance. Read
honestly: **the prose semantic layer is currently sufficient for these cases**,
so formalizing it into governed data is a scale/testability/portability
investment, not a present correctness fix. The right trigger to build the
declarative layer is when this harness (with harder cases or new sources) starts
failing a category — this is that tripwire.

## What a fuller evaluation would measure

The six cases grade one thing well — **answer correctness on high-risk
semantics** — and say nothing about the rest of the system. A production
evaluation would score these dimensions; the table names each, what it checks,
and where it stands so the gap is explicit rather than implied:

| Dimension | What it checks | Status |
|---|---|---|
| **Result-set correctness** | Golden-value equivalence for a question (not SQL string) | ✅ this suite (ground-truth tier), 6 cases |
| **NL→SQL robustness** | Same question, many paraphrases / ambiguous names / unsupported metrics all land correctly | ⚠️ one case each for entity ambiguity, period-mapping, derived metric; **no paraphrase or adversarial breadth** |
| **Route accuracy** | Dispatcher picks the right specialist across question shapes | ❌ not scored — seeded by the `unsupported` case in `tests/test_e2e_smoke.py` |
| **Citation precision** | The cited record actually *supports* the sentence (not just resolves) | ❌ not scored — needs semantic claim↔evidence pairing, not string checks (the honest known-limit in `agents/README.md`) |
| **Citation completeness (recall)** | Every claim that needs a citation has one | ⚠️ counted as a warning-severity lint (`uncited_numeric_sentences`), not scored |
| **Document-retrieval recall** | The chunk that answers a known question is in top-k | ✅ scored — 29-case labelled benchmark (`python -m evals.retrieval`), Recall@5/10 + MRR per strategy; results in `retrieval_report.md`. Has now gated two decisions with data: the hybrid default, and the model2vec → OpenAI embedder upgrade (baseline archived as `retrieval_report_model2vec.md`). |
| **Unsupported-question behaviour** | Off-topic → canned answer, no specialist spend | ✅ asserted end-to-end in `tests/test_e2e_smoke.py` |
| **Chart-to-source consistency** | Every plotted value traces to a cited data point | ✅ structurally enforced (`chart_tool` hydrates from the ledger) **and** asserted end-to-end in `test_e2e_smoke.py` |
| **Latency & token usage** | Cost/time per route, so regressions are visible | ❌ not measured |

Two of these (route sanity, chart-to-source) already have their first automated
assertion in the end-to-end smoke test; the rest are named here so a reviewer
sees a roadmap, not a blind spot. The highest-value next addition is a
**paraphrase/ambiguity set** for NL→SQL robustness — the labelled retrieval
set has since been built (below).

## The document-retrieval benchmark

The second harness in this package compares the three retrieval strategies
(dense / lexical / hybrid — see `../retrieval/README.md`) on 29 labelled
cases across seven categories (semantic paraphrase, exact terminology,
section-scoped, speaker-scoped, metadata-filtered, cross-document,
unsupported). Ground truth is anchor passages located by reading the source
documents and resolved with SQL substring scans — the retriever is never
consulted while labelling, so the labels can't be circular. Methodology and
limitations live in `retrieval_cases.py`; the committed scorecard is
[`retrieval_report.md`](./retrieval_report.md) (+ `.json` with per-case
detail).

```bash
# From be/ — needs Postgres; no LLM. Dense/hybrid embed each case's query,
# so EMBEDDING_PROVIDER=openai (the default) also needs OPENAI_API_KEY
# (29 embedding calls — fractions of a cent). Lexical alone stays key-free.
python -m evals.retrieval                       # print the scorecard
python -m evals.retrieval --strategies lexical  # one strategy
python -m evals.retrieval --write-report        # refresh the committed report
```

Exit code is non-zero if a supported case's ground-truth set resolves empty
(labels drifted from the corpus) — the same build-break convention as golden
SQL drift. Pytest coverage lives in `tests/test_retrieval_strategies.py`
(RRF math, dispatch wiring, live lexical/hybrid queries).

## Extending

Add an `EvalCase` to `evals/cases.py`:
1. Write the natural-language `question` and the golden `reference_sql`
   (returning one row, column `answer`).
2. **Measure** `expected_value` from the database — don't compute it by hand.
3. Set `semantic_rule` to the rule a governed layer would encode.
4. If a semantics-blind agent would take a plausible wrong path, add `trap_sql` /
   `trap_label` / `trap_value` so the report shows the divergence.

Good candidates to add as the system grows: metric-vocabulary aliasing
("profit" → which metric?), cross-company ranking, and — once non-USD data
exists — currency comparability (all rows are USD today, so that guardrail is
currently unexercised).
