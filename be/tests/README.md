# Tests

Tests use [`pytest`](https://docs.pytest.org/). Run them from the `be/` directory:

```bash
pytest                       # the default suite: fast, offline, no API key, no DB
```

## What the suite does — and does not — establish

Be precise about what each test proves. The suite is dominated by **deterministic**
tests that validate application *logic*; a small **live** tier validates the golden
SQL against a real database; and one **opt-in end-to-end** tier is the only thing
that exercises the real model over real infrastructure. Read that way:

| Category | Count* | Needs | What it establishes | What it does **not** |
|---|---:|---|---|---|
| **Pure unit** | 53 | nothing | Logic on inputs: read-only SQL enforcement, SimFin normalization, document chunking, company name-matching, ledger minting, citation validation/repair/lint, chart-spec rendering, eval-case well-formedness + the numeric grader. | Anything involving the DB, the model, or their wiring. |
| **Deterministic behavioural** | 27 | nothing (fakes) | Real code paths driven by **fake chat models** and **monkeypatched I/O boundaries**: the LangChain `create_agent` graphs, the native tool loop, the FastAPI request/response + error-mapping contract, and schema-tool query shaping. Proves the *mechanics* (dispatch, tool-result feedback, repair turn, marker stripping, chart hydration). | Whether the *real* model picks the right tools, or whether retrieval is relevant — the prose here is authored by the test. |
| **Live integration** | 2 | Postgres | Every golden eval query, run through the real read-only SQL tool, returns its **measured** answer, and each documented trap still reproduces. Auto-skips when Postgres isn't reachable. | The agent's behaviour (this tier is ground-truth only). |
| **End-to-end (opt-in)** | 2 | Postgres **+** `OPENAI_API_KEY` **+** `RUN_E2E=1` | `answer_multi()` run against the real model over the real DB: routing behaviour, and the **provenance / chart-to-source invariants on prose the model actually generated**. See `test_e2e_smoke.py`. | Exact figures (asserts structure, not values, to stay non-flaky). |

*Counts are test *functions*; parametrized cases expand the collected item count
(a plain `pytest` run collects ~100 items). "Pure unit" + "deterministic
behavioural" = **80 tests that run offline with no key and no database** — the
default CI-safe suite.

The honest one-line version for a reviewer:

> The ~82 committed tests primarily establish **deterministic application
> behaviour** (fake models drive the real graphs; DB access is monkeypatched).
> Retrieval quality and real model tool-selection are validated separately by an
> **opt-in end-to-end** test against the live stack, because mocked tests cannot
> establish them.

## Running each tier

```bash
pytest                         # pure + deterministic (80) always run; live tier
                               #   auto-runs if Postgres is up, else skips
RUN_E2E=1 pytest -m e2e        # the end-to-end tier — real (paid) model calls,
                               #   needs Postgres up and OPENAI_API_KEY set
python -m evals --agent        # grade the NL->SQL agent against golden answers
                               #   (see evals/README.md) — the accuracy tier
```

The **end-to-end tier is opt-in** (`RUN_E2E=1`): a plain `pytest` never makes a
paid model call even when a key and DB are present, mirroring the eval harness's
`--agent` flag. Where a category is not yet covered by automation (route
accuracy, citation precision/recall, retrieval recall), see the "fuller
evaluation" section of [`../evals/README.md`](../evals/README.md) — that's the
roadmap, named rather than implied.

## Conventions

- Name files `test_*.py`, mirroring the package layout (e.g. `test_sql_tool.py`).
- Keep the default tier free of external services; put anything needing a real
  DB or model behind an availability check (live tier) or the `RUN_E2E` opt-in
  (end-to-end tier), so `pytest` stays fast and hermetic by default.
