# Infrastructure (`/infra`)

Local development infrastructure, run with **Docker Compose**. Today this is a
single Postgres service with **pgvector** for both structured data and the vector
index.

## Services

| Service | Image | Purpose |
| --- | --- | --- |
| `postgres` | `pgvector/pgvector:pg16` | Structured financials **and** the pgvector document index |

Data is persisted in the named volume `pgdata`, so it survives restarts.

## Start / stop

```bash
cd infra
cp .env.example .env          # first time only

docker compose up -d          # start in the background
docker compose ps             # status
docker compose logs -f postgres   # follow logs

docker compose down           # stop, keep data
docker compose down -v        # stop and DELETE data (fresh start)
```

## Connecting the backend

The backend reads `DATABASE_URL` (see [`../be/.env.example`](../be/.env.example)).
With the defaults here that is:

```
postgresql://finapp:finapp@localhost:5432/financial
```

After the container is healthy, apply the schema:

```bash
psql "postgresql://finapp:finapp@localhost:5432/financial" -f ../be/db/schema.sql
```

## Production mapping (future)

This local Compose setup is meant to mirror a cloud deployment so the jump to
production is mostly configuration, not redesign:

| Local (Docker Compose) | Production (AWS) |
| --- | --- |
| Backend container | **ECS / Fargate** services |
| Postgres + pgvector | **RDS / Aurora Postgres** with the pgvector extension |
| Local file storage | **S3** (raw filings, transcripts, cached downloads) |
| `.env` files | **Secrets Manager** / SSM Parameter Store |
| `docker compose logs` | **CloudWatch** + **OpenTelemetry** traces/metrics |

> TODO: Add the backend (and later the frontend) as Compose services once the
> API has a runnable entrypoint beyond the health check, so the whole stack can
> come up with a single `docker compose up`.
