"""Shared PostgreSQL connection settings for retrieval storage operations."""
from __future__ import annotations

import os


def postgres_dsn() -> str:
    """Build a libpq connection string from the documented POSTGRES_* settings."""
    values = {
        "dbname": os.getenv("POSTGRES_DB", "movie_recommender"),
        "user": os.getenv("POSTGRES_USER", "movie_recommender"),
        "password": os.getenv("POSTGRES_PASSWORD", "local_dev_password"),
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
    }
    return " ".join(f"{key}={_quote_conninfo(value)}" for key, value in values.items())


def _quote_conninfo(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
