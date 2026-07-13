# Text-to-SQL evaluation harness

A small, grounded evaluation set for the structured-data query agent. It exists
to answer one question with evidence instead of intuition:

> Which financial semantics does the agent actually get right — and which
> should be promoted from curated prose notes (`tools/schema_tool.py`
> `_TABLE_NOTES`) into a governed, machine-readable model *first*?

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
