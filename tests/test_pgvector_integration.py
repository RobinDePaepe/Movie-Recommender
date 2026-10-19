"""Optional end-to-end checks, isolated in a temporary PostgreSQL schema.

Set POSTGRES_TEST_DSN to run these against a disposable/local test database.
The integration fixture never uses the application's normal database schema.
"""
from __future__ import annotations

import hashlib
import os
import uuid

import numpy as np
import pytest

from movie_recommender.data import movie_database as mdb
from movie_recommender.retrieval.indexing import refresh_retrieval_index
from movie_recommender.retrieval.publishing import publish_retrieval_index
from movie_recommender.retrieval.repository import PgVectorRepository, RetrievalFilters


class IntegrationEmbedder:
    model_id = "test/integration-384"
    model_revision = "v1"
    max_input_tokens = 256

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts):
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            row = np.resize(np.frombuffer(digest, dtype=np.uint8).astype(np.float32) + 1, 384)
            rows.append(row / np.linalg.norm(row))
        return np.asarray(rows, dtype=np.float32)


@pytest.fixture
def isolated_pgvector_dsn():
    base_dsn = os.getenv("POSTGRES_TEST_DSN")
    if not base_dsn:
        pytest.skip("Set POSTGRES_TEST_DSN to run pgvector integration checks")
    psycopg = pytest.importorskip("psycopg")
    from psycopg.conninfo import make_conninfo
    from psycopg import sql

    schema = "retrieval_test_" + uuid.uuid4().hex
    with psycopg.connect(base_dsn) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    isolated_dsn = make_conninfo(base_dsn, options=f"-c search_path={schema},public")
    try:
        yield isolated_dsn
    finally:
        with psycopg.connect(base_dsn) as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


def test_pgvector_publish_search_edit_delete_lifecycle(tmp_path, isolated_pgvector_dsn):
    db = tmp_path / "retrieval.sqlite"
    mdb.init_db(db)
    with mdb.connect(db) as conn:
        movie_id = mdb.upsert_movie(conn, "Velvet Nebula", 2024)
        mdb._upsert_corpus_record(
            conn,
            movie_id_value=movie_id,
            source="letterboxd_diary",
            source_event_id="integration-review-1",
            content_type="review_text",
            text="A dreamlike meditation on memory, silence, and loss.",
            watched_date="2025-04-03",
            rating=4.5,
            rewatch=0,
        )

    embedder = IntegrationEmbedder()
    refresh_retrieval_index(str(db), embedder)
    first_publish = publish_retrieval_index(db, dsn=isolated_pgvector_dsn)
    second_publish = publish_retrieval_index(db, dsn=isolated_pgvector_dsn)
    assert first_publish["published_chunks"] == 1
    assert second_publish["published_chunks"] == 1

    repository = PgVectorRepository(dsn=isolated_pgvector_dsn, embedder=embedder)
    filters = RetrievalFilters(sources=("letterboxd_diary",), content_types=("review_text",))
    lexical = repository.search("dreamlike memory", mode="lexical", filters=filters)
    semantic = repository.search("films about recollection", mode="semantic", filters=filters)
    hybrid = repository.search("dreamlike memory", mode="hybrid", filters=filters)
    assert lexical[0].movie_id == "velvet nebula (2024)"
    assert lexical[0].source_event_id == "integration-review-1"
    assert lexical[0].watched_date == "2025-04-03"
    assert semantic[0].chunk_id == lexical[0].chunk_id
    assert hybrid[0].search_mode == "hybrid"

    with mdb.connect(db) as conn:
        conn.execute(
            "UPDATE retrieval_corpus SET text=? WHERE source_event_id=? AND content_type='review_text'",
            ("A lucid story about starlight and forgotten cities.", "integration-review-1"),
        )
    refresh_retrieval_index(str(db), embedder)
    publish_retrieval_index(db, dsn=isolated_pgvector_dsn)
    assert repository.search("starlight", mode="lexical", filters=filters)
    assert repository.search("dreamlike memory", mode="lexical", filters=filters) == []

    with mdb.connect(db) as conn:
        conn.execute("DELETE FROM retrieval_corpus WHERE source_event_id=?", ("integration-review-1",))
    refresh_retrieval_index(str(db), embedder)
    final = publish_retrieval_index(db, dsn=isolated_pgvector_dsn)
    assert final["published_chunks"] == 0
    assert repository.search("starlight", mode="lexical", filters=filters) == []
