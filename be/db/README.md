# Database

Postgres holds both data domains for the system:

- **Structured** SimFin financials — `companies`, `financials`.
- **Unstructured** document chunks — `documents`, `document_chunks`, indexed
  two ways: [`pgvector`](https://github.com/pgvector/pgvector) embeddings
  (HNSW) for dense similarity, and a generated `content_tsv` tsvector column
  (GIN) for the lexical leg of hybrid retrieval (see
  [`../retrieval/README.md`](../retrieval/README.md)).

See [`schema.sql`](./schema.sql) for the full definition.

## Applying the schema

Start local Postgres via Docker Compose (see [`../../infra`](../../infra)), then:

```bash
psql "$DATABASE_URL" -f be/db/schema.sql
```

`schema.sql` is idempotent (`CREATE ... IF NOT EXISTS`), so it is safe to re-run.

## Notes

- The `document_chunks.embedding` dimension must match the embedding model chosen
  during ingestion — update both together.
- `content_tsv` is `GENERATED ALWAYS AS (to_tsvector('english', content))
  STORED` — it maintains itself on insert/update; adding it to an existing
  database is a table rewrite (~25s at 31k chunks locally). Existing
  databases pick it up by re-running `schema.sql` (the ALTER is
  `IF NOT EXISTS`-idempotent).

> TODO: Adopt a migration tool (e.g. Alembic) once the schema starts changing;
> `schema.sql` is fine for bootstrapping but not for evolving a live database.
