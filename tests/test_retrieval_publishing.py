from __future__ import annotations

import hashlib
import numpy as np
import pytest

from movie_recommender.data import movie_database as mdb
from movie_recommender.retrieval.indexing import refresh_retrieval_index
from movie_recommender.retrieval.publishing import _load_active_index


class FixedEmbedder:
    model_id = "test/fixed-384"
    model_revision = "v1"
    max_input_tokens = 256

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts):
        vectors = []
        for text in texts:
            seed = hashlib.sha256(text.encode("utf-8")).digest()
            row = np.resize(np.frombuffer(seed, dtype=np.uint8).astype(np.float32) + 1, 384)
            vectors.append(row / np.linalg.norm(row))
        return np.asarray(vectors, dtype=np.float32)


def _add_review(db, text="A shimmering film about memory and dreams."):
    mdb.init_db(db)
    with mdb.connect(db) as conn:
        movie_id = mdb.upsert_movie(conn, "Velvet Nebula", 2024)
        mdb._upsert_corpus_record(
            conn,
            movie_id_value=movie_id,
            source="letterboxd_diary",
            source_event_id="review-event-1",
            content_type="review_text",
            text=text,
            watched_date="2025-04-03",
            rating=4.5,
            rewatch=0,
        )


def test_load_active_index_preserves_provenance_and_current_vector(tmp_path):
    db = tmp_path / "staging.sqlite"
    _add_review(db)
    result = refresh_retrieval_index(str(db), FixedEmbedder())

    rows = _load_active_index(db)

    assert result["chunks"] == 1
    assert len(rows) == 1
    row = rows[0]
    assert row["metadata"]["movie_id"] == "velvet nebula (2024)"
    assert row["metadata"]["source"] == "letterboxd_diary"
    assert row["metadata"]["source_event_id"] == "review-event-1"
    assert row["metadata"]["content_type"] == "review_text"
    assert row["metadata"]["watched_date"] == "2025-04-03"
    assert row["metadata"]["rating"] == 4.5
    assert row["vector"].shape == (384,)
    assert row["model_id"] == FixedEmbedder.model_id


def test_load_active_index_rejects_stale_embedding(tmp_path):
    db = tmp_path / "stale.sqlite"
    _add_review(db)
    refresh_retrieval_index(str(db), FixedEmbedder())
    with mdb.connect(db) as conn:
        conn.execute("UPDATE retrieval_embeddings SET source_text_hash=?", ("stale",))

    with pytest.raises(ValueError, match="stale embedding"):
        _load_active_index(db)


def test_load_active_index_rejects_missing_embedding(tmp_path):
    db = tmp_path / "missing.sqlite"
    _add_review(db)
    refresh_retrieval_index(str(db), FixedEmbedder())
    with mdb.connect(db) as conn:
        conn.execute("DELETE FROM retrieval_embeddings")

    with pytest.raises(ValueError, match="only 0 current embeddings"):
        _load_active_index(db)
