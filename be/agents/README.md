# Agents — the structured-data query interface

Turns a natural-language financial question into correct, read-only SQL over
Postgres and returns a **traceable answer**: prose plus every SQL query (and
its rows) and every tool call that produced it.

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

## Design

**Native tool-calling loop, no framework.** `orchestrator.py` implements the
plain OpenAI Chat Completions loop (`finish_reason == "tool_calls"` → execute →
`role: "tool"` replies → repeat). Rationale: the loop is small; every tool call passes
through one choke point where it is validated and recorded (the audit trail is
the product in a financial context); and there is no framework abstraction to
fight when the next requirements (document search, charts) join the same loop.

**Discovery via tools, not schema-in-prompt.** The system prompt contains *no
schema*. The agent discovers tables at runtime (`information_schema`), so a
newly ingested source is queryable the moment its tables land — no prompt or
code change. Source-specific semantics that introspection can't express live
in a curated notes layer (`tools/schema_tool.py:_TABLE_NOTES`) surfaced through
`describe_table`. This split is the "first of many structured sources" design:
introspection scales automatically; notes are additive per source.

## Anticipated pain points and how they're addressed

| # | Pain point | Where it bites | Mitigation |
|---|---|---|---|
| 1 | **Metric names are data, not schema.** In the long/tall `financials` table the model cannot guess `shares_diluted` vs `diluted_shares`. | Silent empty results that look like "no data". | `search_metrics` tool: exact names + statement + company/year coverage, so the agent picks a real metric and knows its coverage before querying. |
| 2 | **Entity resolution.** "Ford" is stored as "Ford Motor"; SimFin has 4,599 companies, most with sparse data. | Wrong or empty company matches; ambiguous tickers. | `resolve_company`: exact-ticker-first fuzzy lookup returning canonical id/name plus data coverage, ranked by how much data exists. |
| 3 | **Source-specific period quirks.** SimFin labels annual balance snapshots `Q4` (not `FY`); quarterly cashflow isn't loaded; fiscal ≠ calendar year. | Confidently wrong period filters (`WHERE fiscal_period='FY'` on balance → 0 rows). | Curated table notes delivered via `describe_table`; the prompt requires respecting notes and disclosing substitutions (e.g. "Q4 snapshot used as annual balance"). |
| 4 | **LLM-generated SQL is untrusted input.** | Writes, DDL, runaway scans. | Defense in depth in `tools/sql_tool.py`: single-SELECT validation + read-only transaction + statement timeout + row cap. `describe_table` validates table names against the live catalog before any interpolation. Production adds a SELECT-only Postgres role. |
| 5 | **Hallucinated tables/columns; bad SQL.** | Query errors mid-conversation. | Errors return to the model as `is_error` tool results — it diagnoses with the discovery tools and retries, bounded by `agent_max_iterations`. |
| 6 | **Traceability.** A number without provenance is useless in finance. | Unverifiable answers. | Every executed query is recorded verbatim with its rows in `AgentAnswer.sql_queries`; every tool call in `tool_calls`. The prompt requires figures to be attributable to numbered queries. |
| 7 | **Context bloat / cost as sources multiply.** A 50-table catalog dumped into the prompt is expensive and stale. | Token cost, cache misses, prompt drift. | Schema is pulled on demand per question; the system prompt is static (cache-friendly); tool results are truncated (row caps, sample-value truncation — a 512-dim embedding never enters context). |
| 8 | **Ambiguity and missing data.** "Annual balance sheet 2025" may not exist yet. | Silent approximation. | Coverage metadata in `resolve_company`/`search_metrics` + a prompt rule: state exactly what's missing; never substitute silently. |

## Extending to the next structured source

1. Ingest into Postgres — the tables appear in `list_tables` automatically.
2. Add a `_TABLE_NOTES` entry per table (join keys, units, quirks).
3. If the source is long/tall, point a metric-catalog query at it (or fold
   into a unified `metrics` view — `search_metrics` is the seam).
4. Cross-source joins hinge on shared entity keys: `companies.id` is the hub
   today (documents already carry `company_name`/`octus_company_id`; mapping
   them onto `companies` is the planned join — see db/schema.sql TODO).

Nothing in the loop, prompt, or `/ask` endpoint changes.

## Files

- `orchestrator.py` — the loop: `answer(question) -> AgentAnswer`.
- `prompts.py` — static, source-agnostic system prompt.
- `schemas.py` — `AgentAnswer` / `SqlQueryRecord` / `ToolCallRecord`.
- `../tools/schema_tool.py` — discovery tools + curated notes.
- `../tools/sql_tool.py` — read-only SQL execution (defense in depth).

## Running it

```bash
# needs OPENAI_API_KEY in be/.env (see .env.example) and Postgres up
uvicorn app.main:app --reload --port 8000
curl -s localhost:8000/ask -X POST -H 'content-type: application/json' \
     -d '{"question": "Compare Ford and Delta revenue for fiscal 2024."}' | jq

# or from Python
python -c "from agents.orchestrator import answer; print(answer('...').answer)"
```

Tests: `pytest tests/test_agent_loop.py tests/test_schema_tool.py` (no API key
or database needed — the client is faked and DB calls are monkeypatched).
