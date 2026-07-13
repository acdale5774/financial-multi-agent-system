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

    # Embeddings. model2vec static embeddings run locally with no API key or
    # GPU — chosen because torch/onnxruntime don't install on this dev machine
    # (Intel Mac + Python 3.14). Swap the model (and matching dimension +
    # db/schema.sql vector size) to upgrade; re-index afterwards.
    embedding_model: str = "minishlab/potion-retrieval-32M"
    embedding_dimension: int = 512

    # Agent layer (structured-data query interface). If the key is empty, the
    # OpenAI SDK's own resolution (OPENAI_API_KEY env var) is used instead.
    openai_api_key: str = ""
    agent_model: str = "gpt-5.5"
    agent_max_iterations: int = 15


# Import this singleton wherever settings are needed.
settings = Settings()
