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


# Import this singleton wherever settings are needed.
settings = Settings()
