"""Publish the active SQLite retrieval index to PostgreSQL/pgvector."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from movie_recommender.data import movie_database as mdb
from movie_recommender.retrieval.postgres import postgres_dsn

MIGRATION_DIR = Path(__file__).resolve().parents[3] / "migrations" / "postgres"
VECTOR_DIMENSION = 384
_MIGRATION_NAME = re.compile(r"^[0-9]+_[a-z0-9_]+\.sql$")


def _load_active_index(db_path: str | Path) -> list[dict[str, Any]]:
    """Read chunks and their current embeddings in one SQLite snapshot."""
    mdb.init_db(db_path)
    with mdb.connect(db_path) as conn:
        chunk_count = conn.execute("SELECT count(*) FROM retrieval_chunks").fetchone()[0]
        rows = conn.execute(
            """SELECT c.chunk_id,c.record_id,c.chunk_index,c.chunker_version,c.text,c.text_hash,
                      c.metadata_json,e.model_id,e.model_revision,e.vector_dimension,e.vector,
                      e.source_text_hash
               FROM retrieval_chunks c JOIN retrieval_embeddings e USING(chunk_id)
               ORDER BY c.chunk_id"""
        ).fetchall()
    if len(rows) != chunk_count:
        raise ValueError(
            f"SQLite index has {chunk_count} chunks but only {len(rows)} current embeddings; "
            "rebuild the SQLite index before publishing."
        )

    result: list[dict[str, Any]] = []
    model_pairs: set[tuple[str, str]] = set()
    for row in rows:
        if row["source_text_hash"] != row["text_hash"]:
            raise ValueError(f"Chunk {row['chunk_id']} has a stale embedding; rebuild the SQLite index first.")
        dimension = int(row["vector_dimension"])
        if dimension != VECTOR_DIMENSION:
            raise ValueError(
                f"Chunk {row['chunk_id']} has {dimension} dimensions; the PostgreSQL schema expects {VECTOR_DIMENSION}."
            )
        vector = np.frombuffer(row["vector"], dtype=np.float32)
        if vector.size != dimension or not np.isfinite(vector).all():
            raise ValueError(f"Chunk {row['chunk_id']} has an invalid stored vector.")
        metadata = json.loads(row["metadata_json"])
        model_pairs.add((row["model_id"], row["model_revision"]))
        result.append({
            "chunk_id": row["chunk_id"], "record_id": row["record_id"],
            "chunk_index": int(row["chunk_index"]), "chunker_version": row["chunker_version"],
            "text": row["text"], "text_hash": row["text_hash"], "metadata": metadata,
            "model_id": row["model_id"], "model_revision": row["model_revision"],
            "vector": vector,
        })
    if len(model_pairs) > 1:
        raise ValueError("SQLite retrieval index contains multiple embedding model versions; rebuild it before publishing.")
    return result


def _apply_migrations(conn: Any) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS schema_migrations (
               migration_name text PRIMARY KEY,
               applied_at timestamptz NOT NULL DEFAULT now()
           )"""
    )
    for path in sorted(MIGRATION_DIR.glob("*.sql")):
        if not _MIGRATION_NAME.fullmatch(path.name):
            raise ValueError(f"Invalid PostgreSQL migration filename: {path.name}")
        exists = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE migration_name=%s", (path.name,)
        ).fetchone()
        if exists:
            continue
        conn.execute(path.read_text(encoding="utf-8"))
        conn.execute("INSERT INTO schema_migrations (migration_name) VALUES (%s)", (path.name,))


def publish_retrieval_index(
    db_path: str | Path = mdb.DB_PATH,
    *,
    dsn: str | None = None,
    batch_size: int = 500,
) -> dict[str, int]:
    """Idempotently mirror the active SQLite index, including edits and deletions."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    rows = _load_active_index(db_path)
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError as exc:
        raise RuntimeError("Publishing requires psycopg and pgvector. Install the project dependencies first.") from exc

    inserted_or_updated = 0
    with psycopg.connect(dsn or postgres_dsn()) as conn:
        register_vector(conn)
        with conn.transaction():
            _apply_migrations(conn)
            conn.execute(
                "CREATE TEMP TABLE publish_chunk_ids (chunk_id text PRIMARY KEY) ON COMMIT DROP"
            )
            cursor = conn.cursor()
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                values = []
                for item in batch:
                    meta = item["metadata"]
                    rewatch = meta.get("rewatch")
                    values.append((
                        item["chunk_id"], item["record_id"], meta["movie_id"], meta["source"],
                        meta["source_event_id"], meta["content_type"], item["chunk_index"],
                        item["chunker_version"], item["text"], item["text_hash"],
                        meta.get("movie_name"), meta.get("movie_year"), meta.get("watched_date"),
                        meta.get("rating"), bool(rewatch) if rewatch is not None else None, meta.get("tmdb_id"),
                        meta.get("letterboxd_uri"), item["model_id"], item["model_revision"],
                        item["vector"],
                    ))
                if not values:
                    continue
                cursor.executemany(
                    "INSERT INTO publish_chunk_ids (chunk_id) VALUES (%s) ON CONFLICT DO NOTHING",
                    [(item["chunk_id"],) for item in batch],
                )
                cursor.executemany(
                    """INSERT INTO retrieval_chunks
                       (chunk_id,record_id,movie_id,source,source_event_id,content_type,chunk_index,
                        chunker_version,text,text_hash,movie_name,movie_year,watched_date,rating,rewatch,
                        tmdb_id,letterboxd_uri,model_id,model_revision,embedding)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (chunk_id) DO UPDATE SET
                         record_id=excluded.record_id,movie_id=excluded.movie_id,source=excluded.source,
                         source_event_id=excluded.source_event_id,content_type=excluded.content_type,
                         chunk_index=excluded.chunk_index,chunker_version=excluded.chunker_version,
                         text=excluded.text,text_hash=excluded.text_hash,movie_name=excluded.movie_name,
                         movie_year=excluded.movie_year,watched_date=excluded.watched_date,
                         rating=excluded.rating,rewatch=excluded.rewatch,tmdb_id=excluded.tmdb_id,
                         letterboxd_uri=excluded.letterboxd_uri,model_id=excluded.model_id,
                         model_revision=excluded.model_revision,embedding=excluded.embedding,
                         updated_at=now()
                       WHERE (retrieval_chunks.record_id,retrieval_chunks.movie_id,retrieval_chunks.source,
                         retrieval_chunks.source_event_id,retrieval_chunks.content_type,retrieval_chunks.chunk_index,
                         retrieval_chunks.chunker_version,retrieval_chunks.text,retrieval_chunks.text_hash,
                         retrieval_chunks.movie_name,retrieval_chunks.movie_year,retrieval_chunks.watched_date,
                         retrieval_chunks.rating,retrieval_chunks.rewatch,retrieval_chunks.tmdb_id,
                         retrieval_chunks.letterboxd_uri,retrieval_chunks.model_id,retrieval_chunks.model_revision,
                         retrieval_chunks.embedding) IS DISTINCT FROM
                        (excluded.record_id,excluded.movie_id,excluded.source,excluded.source_event_id,
                         excluded.content_type,excluded.chunk_index,excluded.chunker_version,excluded.text,
                         excluded.text_hash,excluded.movie_name,excluded.movie_year,excluded.watched_date,
                         excluded.rating,excluded.rewatch,excluded.tmdb_id,excluded.letterboxd_uri,
                         excluded.model_id,excluded.model_revision,excluded.embedding)""",
                    values,
                )
                inserted_or_updated += len(batch)
            conn.execute(
                "DELETE FROM retrieval_chunks c WHERE NOT EXISTS "
                "(SELECT 1 FROM publish_chunk_ids p WHERE p.chunk_id=c.chunk_id)"
            )
            count = conn.execute("SELECT count(*) FROM retrieval_chunks").fetchone()[0]
    return {"source_chunks": len(rows), "published_chunks": int(count), "processed_chunks": inserted_or_updated}
