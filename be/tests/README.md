# Tests

Tests use [`pytest`](https://docs.pytest.org/). Run them from the `be/` directory:

```bash
pytest
```

## What to cover first

As functionality lands, prioritize the pieces where correctness and safety matter
most:

- **`tools/sql_tool.py`** — read-only enforcement: reject DML/DDL, multiple
  statements, and write-CTEs; confirm SELECTs succeed and `max_rows` is honored.
- **`ingestion/simfin_ingest.py`** — `normalize_simfin_data` field mapping, type
  coercion, and handling of incomplete rows.
- **`ingestion/document_ingest.py`** — `chunk_document` boundary behavior
  (chunk sizing and overlap).
- **`tools/chart_tool.py`** — `build_chart` produces a valid spec and fails
  clearly when referenced fields are missing.

## Conventions

- Name files `test_*.py`, mirroring the package layout (e.g. `test_sql_tool.py`).
- Keep unit tests free of external services; use a disposable local Postgres
  (the Docker Compose instance) for the few integration tests that need a DB.

> TODO: Add a `conftest.py` with fixtures (temp DB, sample SimFin payloads,
> sample document text) once the first real tests are written.
