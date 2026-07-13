"""Database package: connection helpers plus the SQL schema.

`schema.sql` is the source of truth for the relational + pgvector layout;
`connection.py` provides the psycopg connection used by ingestion and tools.
"""

from .connection import apply_schema, get_connection

__all__ = ["apply_schema", "get_connection"]
