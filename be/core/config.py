"""Application settings, loaded from the environment and an optional `.env` file.

Uses pydantic-settings so a local `.env` (see `.env.example`) is picked up
automatically in development, while real deployments inject the same names as
environment variables / secrets.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Postgres (pgvector-enabled) connection string.
    database_url: str = "postgresql://finapp:finapp@localhost:5432/financial"

    # SimFin bulk-dataset ingestion. NOTE: the old "free" key now returns HTTP 401
    # on the bulk-download endpoint — a real API key from simfin.com is required.
    # Set SIMFIN_API_KEY in `.env`.
    simfin_api_key: str = ""
    simfin_data_dir: str = "./_simfin_cache"
    simfin_market: str = "us"

    # Document ingestion (SEC filings + transcripts).
    documents_data_dir: str = "../case_study_data"

    # Embeddings. Default: OpenAI text-embedding-3-small (needs OPENAI_API_KEY,
    # which the agent layer already requires). Upgraded from local model2vec
    # static embeddings after the retrieval benchmark measured dense recall as
    # the weakest link (evals/retrieval_report_model2vec.md is the before
    # scorecard). Offline fallback: EMBEDDING_PROVIDER=model2vec with
    # EMBEDDING_MODEL=minishlab/potion-retrieval-32M and EMBEDDING_DIMENSION=512.
    # Any change here means updating the vector(N) size in db/schema.sql,
    # re-applying it, and re-embedding: python -m ingestion.reembed
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Document retrieval strategy: 'dense' | 'lexical' | 'hybrid'. The default
    # is chosen from the labeled benchmark (`python -m evals.retrieval`; report
    # in evals/retrieval_report.md) — change it only with benchmark evidence.
    # 2026-07 result (OpenAI text-embedding-3-small): hybrid 96% R@10 / 0.779
    # MRR vs dense 88% / 0.754 and lexical 81% / 0.653 — hybrid beats both
    # legs on every overall metric. (Under the earlier model2vec embedder the
    # same call was a hedge — see evals/retrieval_report_model2vec.md.)
    retrieval_strategy: str = "hybrid"
    # Hybrid-only tuning: how many candidates each leg contributes to fusion,
    # and the RRF constant (60 is the conventional default from the RRF paper;
    # larger flattens the rank weighting).
    retrieval_leg_k: int = 30
    retrieval_rrf_k: int = 60

    # Agent layer (structured-data query interface). If the key is empty, the
    # OpenAI SDK's own resolution (OPENAI_API_KEY env var) is used instead.
    openai_api_key: str = ""
    agent_model: str = "gpt-5.5"
    agent_max_iterations: int = 15


# Import this singleton wherever settings are needed.
settings = Settings()
