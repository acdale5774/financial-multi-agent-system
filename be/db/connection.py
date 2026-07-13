"""Postgres connection helpers.

A thin wrapper over psycopg so the rest of the backend never hard-codes a
connection string and always gets dict-shaped rows.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from core.config import settings

# Location of the canonical schema file (this package's schema.sql).
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def get_connection(
    database_url: str | None = None,
    *,
    row_factory=dict_row,
) -> psycopg.Connection:
    """Open a new Postgres connection.

    Args:
        database_url: Override connection string; defaults to `settings.database_url`.
        row_factory: psycopg row factory; defaults to dict rows.

    Returns:
        An open `psycopg.Connection` (caller is responsible for closing, e.g.
        by using it as a context manager).
    """
    return psycopg.connect(database_url or settings.database_url, row_factory=row_factory)


def apply_schema(database_url: str | None = None) -> None:
    """Apply `schema.sql` to bootstrap tables and the pgvector extension.

    The schema is idempotent (`CREATE ... IF NOT EXISTS`), so this is safe to run
    repeatedly. Equivalent to `psql "$DATABASE_URL" -f db/schema.sql`.
    """
    sql = SCHEMA_PATH.read_text()
    with get_connection(database_url) as conn:
        # Full file may contain several statements; send it via the simple
        # protocol (no bound params) so all statements run.
        conn.execute(sql)
        conn.commit()
