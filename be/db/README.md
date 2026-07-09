# Database

Postgres holds both data domains for the system:

- **Structured** SimFin financials — `companies`, `financials`.
- **Unstructured** document chunks with embeddings — `documents`,
  `document_chunks` (uses the [`pgvector`](https://github.com/pgvector/pgvector)
  extension for similarity search).

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
- The ANN (approximate nearest neighbor) index on `embedding` is intentionally
  left as a TODO until there is data to tune it against.

> TODO: Adopt a migration tool (e.g. Alembic) once the schema starts changing;
> `schema.sql` is fine for bootstrapping but not for evolving a live database.
