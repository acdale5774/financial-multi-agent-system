# Agents

Two entry points, two generations of the same philosophy:

- **`POST /agent/ask`** — the **multi-agent system** (requirement 3): routes a
  question to a specialist agent, returns prose where **every claim carries a
  citation** to a SimFin data point (ticker + fiscal period) or a document
  chunk (document + section), plus rendered charts and the full evidence trace.
- **`POST /ask`** — the single-agent **structured-data query interface**
  (requirement 2), kept untouched as a baseline and live fallback.

## Requirement 3 — multi-agent architecture

```
POST /agent/ask {"question": ...}
   │
   ├─ 1. DISPATCH (one structured-output LLM call; any failure → "hybrid")
   │       route ∈ {quant, docs, hybrid, unsupported}
   │
   ├─ 2. ONE SPECIALIST (LangChain create_agent tool loop, gpt-5.5)
   │       quant   → list_tables · describe_table · search_metrics ·
   │                 resolve_company · query_financials · create_chart
   │       docs    → document_coverage · search_documents · resolve_company
   │       hybrid  → union of both (genuinely cross-domain questions)
   │
   │       every evidence-bearing tool writes to the per-request
   │       ┌──────────────────────────────────────────────────────┐
   │       │ EVIDENCE LEDGER — citation ids minted BY CODE:       │
   │       │  S# = SimFin data point (ticker, metric, FY/Q, value)│
   │       │  D# = document chunk (doc, section, snippet)         │
   │       │  C# = rendered chart (values hydrated from S# ids)   │
   │       └──────────────────────────────────────────────────────┘
   │
   └─ 3. VALIDATE (deterministic): every [S#]/[D#] marker must resolve to a
          ledger record → one repair turn → strip-and-warn. Lints (warning
          severity): uncited numeric sentences; cited-value mismatches.
   <── MultiAgentAnswer {answer, citations[], charts[], validation,
                         sql_queries[], searches[], route, route_reason}
```

**Framework: LangChain** (`create_agent`, LangChain 1.3 / LangGraph 1.2). Of
the three allowed frameworks it is the only one that is first-class on the
OpenAI models this project runs on (Claude Agent SDK drives Claude; Google ADK
is Gemini-first). Specialists are compiled LangGraph graphs; the outer flow
(dispatch → specialist → validate) is four deterministic steps of plain
Python — a wrapper StateGraph would add ceremony, not capability.

### Routing logic

The dispatcher is a **classifier, not a supervisor**: one
`with_structured_output(Route)` call against a rubric prompt with the five
case-study question shapes as few-shots. Deterministic properties:

- Routing can never fail a request — any exception falls back to `hybrid`,
  whose toolset is a superset of both specialists.
- `unsupported` (off-topic questions) short-circuits to a canned capabilities
  answer without spending a specialist run.
- The route and its reason are echoed in every response for transparency.

Why not a supervisor that delegates to specialists-as-tools? It was the
runner-up design. A supervisor re-summarizes specialist output and roughly
doubles-to-triples LLM hops per request; with a reasoning model that is the
difference between a ~20–60 s answer and a multi-minute one, and every added
hand-off is a place for a number to mutate. At this question complexity the
classifier routes just as well, and cross-domain synthesis happens inside one
hybrid context instead of across a lossy hand-off. The seam is documented:
when domains multiply, the dispatcher grows routes; if inter-specialist
coordination is ever truly needed, the hybrid slot is where a supervisor goes.

### Citations, end to end

1. **Capture (code, not model).** `query_financials` executes the SQL, then
   the ledger mints one `SimFinCitation` per *numeric column* of every row
   that projects the citation keys (ticker/company + fiscal_year +
   fiscal_period) — a pivot row with revenue, net_income and computed EPS
   yields three ids. Rows without keys (aggregates) degrade to one
   query-granularity citation and the tool result tells the model how to do
   better. `search_documents` mints one `DocumentCitation` per chunk with a
   code-captured snippet. The ids arrive in the tool result *next to the
   values* (`"cites": {"revenue": "S3"}`), so citing correctly is the path of
   least resistance.
2. **Reference.** Specialist prompts require a marker per claim
   (`... $185.0B in FY2024 [S3].`). The model can only repeat ids it saw —
   fabricating a citation *record* is structurally impossible.
3. **Enforce (deterministic).** Markers that don't resolve to the ledger
   (S/D/C/T — lowercase and range forms are normalized first) trigger one
   repair turn (the corrective message lists the real inventory); whatever is
   still invalid is stripped and reported in
   `validation.invalid_markers_stripped`, and markdown images pointing at
   chart URLs the ledger never rendered are removed. The endpoint never 500s
   over citations — it degrades honestly.
4. **Lint (warning severity, by design; hardened after an adversarial
   review).** Sentences stating figures without a marker are counted; for
   marked sentences, at least half of the stated numbers must be found in
   the cited evidence, where matching is *unit-aware* ("$394.3 million"
   cannot pass against a cited 394.3 billion), *quantization-aware* ("1.8B"
   legitimately stands for 1.831B), and evidence values come from the cited
   data point, the cited query's actual rows (query-granularity citations),
   or the cited document chunk's text (a number "cited" to a chunk that
   never states it is flagged). These stay warnings because sentence
   splitting and number matching are heuristics — hard-failing them would
   strip true claims (e.g. "3.0× higher", a derived comparison of two cited
   figures).

Computed figures (EPS, growth) are minted from the rows of the query that
computed them, so the formula is auditable in the verbatim SQL at
`citations[].sql_index` — stronger provenance than a model-declared formula.

### Charts and tables (requirement 4)

`create_chart` takes x-labels and **citation ids**, never numbers: y-values
are **hydrated from the ledger**, so a plotted value that wasn't cited
evidence is structurally impossible — and the tool rejects a series that
repeats a citation id or an x-label that doesn't name the cited point's
fiscal period and year (a "quarterly 2025" axis can't be faked from one
FY2022 data point). `create_table` applies the same rule to structured
visuals: the model supplies column headers, row labels, and one citation id
per cell; the values and the markdown (markers included) are generated by
code, and the typed `TableRecord` rides in `tables[]` for the frontend to
render with per-cell evidence chips. Rendering is pure code
(`tools/chart_tool.py`, Agg backend, `Figure` directly — pyplot's global
state is not thread-safe under concurrent requests). The PNG is served at
`/charts/…`; the declarative spec rides along in `charts[].spec`. The
specialist prompts call for a chart or table proactively whenever a trend,
comparison, or collation would be clearer visually.

### How the five example questions flow

| Question | Route | What happens |
|---|---|---|
| Collate Revenue/Net Income/EPS for X, last 4 quarters + plot EPS | quant | one pivot query projecting citation keys (EPS computed as `net_income / NULLIF(shares_diluted,0)`); `create_table` hydrates the collation with a marker per cell; `create_chart` plots the cited EPS ids |
| Fastest YoY revenue growth | quant | self-join/window over FY revenue, growth computed in SQL (so it's minted + citable), currency-scoped, revenue floor against penny-denominator noise |
| Plot revenue for X, last 3 quarters | quant | resolve → one quarterly query → chart from the three cited ids |
| Trends/risks across [sector] disclosures | docs | `document_coverage` scopes the corpus (12 companies), filtered searches (`sub_industry`, `qa_role='Answer'`, risk-factor sections), themes attributed per company with [D#] |
| List companies with CEO, Sector, recent Annual Revenue | quant | aggregate + LIMIT (4,599 rows aren't enumerable), Sector from the backfilled column; **CEO honestly reported unavailable** (SimFin has no officer data — the curated notes say so, and inventing one would fail citation) |

### Tradeoffs

| Decision | Road not taken | Why |
|---|---|---|
| Classifier dispatch + one specialist per request | Supervisor with specialists-as-tools | 2 LLM roles vs 3+ per request: latency and hand-off mutation risk beat delegation flexibility at this scale; hybrid superset makes misroutes fail soft |
| Hybrid = union toolset in one context | docs+quant fan-out with an LLM composer | cross-domain synthesis in one context; a composer is a third hop and a lossy hand-off. Cost: the union prompt fattens as domains grow — the documented seam for a future supervisor |
| Citation ids minted by code at the tool boundary | Model-emitted citation objects (`response_format`) | fabricated records impossible; validation is a dict lookup; no structured-output/tool-loop interplay. Cost: model can still put a *valid* id on a *wrong* sentence — watched by the value-mismatch lint, disclosed here |
| Coverage lints at warning severity | Repair-triggering hard failures | heuristics false-positive ("top 10", derived ratios); hard failure would strip true claims mid-demo. Validation block keeps the signal visible |
| One repair turn, then strip-and-warn | Regenerate-until-valid loop | bounded latency, endpoint never 500s; a stripped marker is reported, not hidden |
| Chart y-values hydrated from ledger | Trusting model-passed numbers (or LLM plotting code) | closes the wrong-chart failure entirely; no code-execution surface; deterministic renders can't flake |
| Keep req-2 loop at `/ask` untouched | Absorb it into the quant specialist | zero regression to a live-verified deliverable; preserves the native-loop-vs-framework comparison both are graded on |
| EPS/growth computed in SQL, provenance = verbatim query | Materialized `eps` rows; model-declared formula strings | no derived-data staleness; the formula is exactly where the auditor looks (`sql_index`) |

### Extending to source N+1

Unchanged from requirement 2, now one level up: ingest the tables and they are
discoverable (`list_tables` is live introspection); add a `_TABLE_NOTES` entry
for the quirks; the quant specialist can use them immediately — no prompt,
graph, or endpoint changes. A genuinely new *modality* (e.g. a market-data
API) becomes either a new tool on an existing specialist or a new route +
specialist; the ledger only needs a new citation type if the evidence shape is
new. `companies.id`/ticker is the cross-source join key (documents are mapped
via `ingestion/enrich_companies.py`).

### Known limits (also the honest-interview list)

- A valid citation id on a sentence whose numbers happen to match *other*
  cited values (swapped attribution) passes the lints; closing that needs
  semantic claim–evidence pairing, not string checks.
- The docs corpus is 12 companies — sector questions are scoped to it and the
  answers say so; SimFin has no CEO/officer data.
- `_charts/` accumulates PNGs (fine locally; a real deployment adds a sweep).

---

## Requirement 2 — the structured-data query interface (`POST /ask`)

Turns a natural-language financial question into read-only SQL over Postgres
and returns a **traceable answer**: prose plus every SQL query (and its rows)
and every tool call that produced it.

```
POST /ask {"question": "How did Ford's quarterly revenue trend through 2024?"}
   └─> agents/orchestrator.answer()
         LLM (tool-calling loop)
           ├─ list_tables / describe_table   ← live schema + curated notes
           ├─ resolve_company("Ford")        ← entity resolution + coverage
           ├─ search_metrics("revenue")      ← metric catalog (long/tall)
           └─ run_sql("SELECT ...")          ← read-only, capped, recorded
   <── AgentAnswer {answer, sql_queries[], tool_calls[], model, iterations}
```

**Native tool-calling loop, no framework.** `orchestrator.py` implements the
plain OpenAI Chat Completions loop (`finish_reason == "tool_calls"` → execute →
`role: "tool"` replies → repeat). Rationale: the loop is small; every tool call
passes through one choke point where it is validated and recorded (the audit
trail is the product in a financial context). Requirement 3 then deliberately
rebuilds this pattern on LangChain — the two directly compare a 60-line native
loop against a framework, and the framework's testability (fake chat models
driving real graphs) and multi-agent wiring is what justified it at that scale.

**Discovery via tools, not schema-in-prompt.** The system prompt contains *no
schema*. The agent discovers tables at runtime (`information_schema`), so a
newly ingested source is queryable the moment its tables land — no prompt or
code change. Source-specific semantics that introspection can't express live
in a curated notes layer (`tools/schema_tool.py:_TABLE_NOTES`) surfaced through
`describe_table`. This split is the "first of many structured sources" design:
introspection scales automatically; notes are additive per source.

### Anticipated pain points and how they're addressed

| # | Pain point | Where it bites | Mitigation |
|---|---|---|---|
| 1 | **Metric names are data, not schema.** In the long/tall `financials` table the model cannot guess `shares_diluted` vs `diluted_shares`. | Silent empty results that look like "no data". | `search_metrics` tool: exact names + statement + company/year coverage, so the agent picks a real metric and knows its coverage before querying. |
| 2 | **Entity resolution.** "Ford" is stored as "FORD MOTOR CO"; ticker `FORD` is Forward Industries. SimFin has 4,599 companies, most with sparse data. | Wrong or empty company matches; ambiguous tickers. | `resolve_company`: exact-ticker-first fuzzy lookup returning canonical id/name plus data coverage, ranked by how much data exists. |
| 3 | **Source-specific period quirks.** SimFin labels annual balance snapshots `Q4` (not `FY`); quarterly cashflow isn't loaded; fiscal ≠ calendar year; no `eps` metric; currency varies per row. | Confidently wrong period filters (`WHERE fiscal_period='FY'` on balance → 0 rows); apples-to-oranges rankings. | Curated table notes delivered via `describe_table`; the prompt requires respecting notes and disclosing substitutions. |
| 4 | **LLM-generated SQL is untrusted input.** | Writes, DDL, runaway scans. | Defense in depth in `tools/sql_tool.py`: single-SELECT validation + read-only transaction + statement timeout + row cap. `describe_table` validates table names against the live catalog before any interpolation. Production adds a SELECT-only Postgres role. |
| 5 | **Hallucinated tables/columns; bad SQL.** | Query errors mid-conversation. | Errors return to the model as error tool results — it diagnoses with the discovery tools and retries, bounded by `agent_max_iterations`. |
| 6 | **Traceability.** A number without provenance is useless in finance. | Unverifiable answers. | Every executed query is recorded verbatim with its rows; requirement 3 upgrades this to per-claim citations. |
| 7 | **Context bloat / cost as sources multiply.** A 50-table catalog dumped into the prompt is expensive and stale. | Token cost, cache misses, prompt drift. | Schema is pulled on demand per question; the system prompt is static (cache-friendly); tool results are truncated (row caps, sample-value truncation — a 512-dim embedding never enters context). |
| 8 | **Ambiguity and missing data.** "Annual balance sheet 2025" may not exist yet. | Silent approximation. | Coverage metadata in `resolve_company`/`search_metrics`/`document_coverage` + a prompt rule: state exactly what's missing; never substitute silently. |

## Files

- `multi_agent.py` — requirement 3 pipeline: `answer_multi(question) -> MultiAgentAnswer`.
- `lc_tools.py` — per-request LangChain tool closures over the ledger.
- `ledger.py` — evidence ledger; mints S#/D#/C# citation records.
- `citations.py` — deterministic marker validation, repair, lints.
- `prompts.py` — router + specialist prompts (source-agnostic) and the
  requirement-2 system prompt.
- `schemas.py` — `AgentAnswer`, `MultiAgentAnswer`, citation models.
- `orchestrator.py` — requirement 2's native loop: `answer(question) -> AgentAnswer`.
- `../tools/schema_tool.py` — schema discovery + curated notes (shared).
- `../tools/document_search_tool.py` — filtered vector search + `document_coverage`.
- `../tools/chart_tool.py` — deterministic PNG rendering.
- `../tools/sql_tool.py` — read-only SQL execution (defense in depth).

## Running it

```bash
# needs OPENAI_API_KEY in be/.env (see .env.example) and Postgres up
uvicorn app.main:app --reload --port 8000

curl -s localhost:8000/agent/ask -X POST -H 'content-type: application/json' \
     -d '{"question": "Plot the revenue for Ford over the last 3 quarters."}' | jq
# answer prose + citations[] + charts[] (PNG at localhost:8000/charts/...)

curl -s localhost:8000/ask -X POST -H 'content-type: application/json' \
     -d '{"question": "Compare Ford and Delta revenue for fiscal 2024."}' | jq
```

Tests: `pytest tests/` (no API key or database needed — LangChain fake chat
models drive the real graphs; DB calls are monkeypatched).
