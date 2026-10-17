from __future__ import annotations

import hashlib
import json

import numpy as np

from movie_recommender.data import movie_database as mdb
from movie_recommender.retrieval.indexing import build_chunks, chunk_text, refresh_retrieval_index


class FakeEmbedder:
    model_id = "test/fake-embedder"
    max_input_tokens = 256

    def __init__(self, revision: str = "v1", dimensions: int = 4):
        self.model_revision = revision
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts):
        batch = list(texts)
        self.calls.append(batch)
        rows = []
        for text in batch:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rows.append([float(digest[i]) + 1.0 for i in range(self.dimensions)])
        return np.asarray(rows, dtype=np.float32)


def _add_record(db, *, name, year, source, event, content_type, text,
                watched_date=None, rating=None, rewatch=None, tmdb_id=None):
    mdb.init_db(db)
    with mdb.connect(db) as conn:
        movie_id = mdb.upsert_movie(conn, name, year, tmdb_id=tmdb_id)
        mdb._upsert_corpus_record(
            conn,
            movie_id_value=movie_id,
            source=source,
            source_event_id=event,
            content_type=content_type,
            text=text,
            watched_date=watched_date,
            rating=rating,
            rewatch=rewatch,
            tmdb_id=tmdb_id,
        )
        record_id = conn.execute(
            "SELECT record_id FROM retrieval_corpus WHERE source=? AND source_event_id=? AND content_type=?",
            (source, event, content_type),
        ).fetchone()[0]
    return record_id


def test_chunk_text_keeps_short_text_and_overlaps_sentence_chunks():
    embedder = FakeEmbedder()
    short = "One short review."
    chunks, version = chunk_text(short, embedder)
    assert chunks == [short]
    assert "sentence-boundary-v1" in version

    long_text = " ".join(f"word{i}." for i in range(450))
    long_chunks, _ = chunk_text(long_text, embedder)
    assert len(long_chunks) >= 3
    assert all(embedder.count_tokens(chunk) <= 200 for chunk in long_chunks)
    assert long_chunks[0].split()[-32:] == long_chunks[1].split()[:32]


def test_build_chunks_preserves_typed_provenance_and_skips_empty_diary():
    embedder = FakeEmbedder()
    records = [
        {"record_id": "review-id", "movie_id": "film (2020)", "movie_name": "Film", "movie_year": 2020,
         "source": "letterboxd_diary", "source_event_id": "event-1", "content_type": "review_text",
         "text": "A short review.", "watched_date": "2020-01-02", "rating": 4.5, "rewatch": 1,
         "tmdb_id": None, "letterboxd_uri": "https://example.invalid/film"},
        {"record_id": "event-id", "movie_id": "film (2020)", "source": "letterboxd_diary",
         "source_event_id": "event-1", "content_type": "diary_event", "text": ""},
    ]
    chunks = build_chunks(records, embedder)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == build_chunks(records, embedder)[0].chunk_id
    metadata = json.loads(chunks[0].metadata_json)
    assert metadata["movie_id"] == "film (2020)"
    assert metadata["source_event_id"] == "event-1"
    assert metadata["watched_date"] == "2020-01-02"
    assert metadata["rating"] == 4.5
    assert metadata["rewatch"] == 1
    assert metadata["tmdb_id"] is None


def test_refresh_reuses_updates_rebuilds_and_cascades_stale_embeddings(tmp_path):
    db = tmp_path / "retrieval.sqlite"
    review_text = " ".join(f"review{i}." for i in range(450))
    review_id = _add_record(
        db, name="Film", year=2020, source="letterboxd_diary", event="diary-1",
        content_type="review_text", text=review_text, watched_date="2020-01-02", rating=4.5, rewatch=1,
    )
    _add_record(
        db, name="Film", year=2020, source="tmdb", event="tmdb:film:overview",
        content_type="tmdb_overview", text="A compact plot summary.", tmdb_id=123,
    )
    _add_record(
        db, name="Film", year=2020, source="letterboxd_diary", event="diary-1",
        content_type="diary_event", text="", watched_date="2020-01-02", rating=4.5, rewatch=1,
    )

    first_embedder = FakeEmbedder("v1")
    first = refresh_retrieval_index(str(db), first_embedder)
    assert first["chunks"] >= 4
    assert first["embeddings"] == first["chunks"]
    assert first["embedded"] == first["chunks"]
    assert sum(len(batch) for batch in first_embedder.calls) == first["chunks"]

    with mdb.connect(db) as conn:
        chunks = conn.execute(
            "SELECT chunk_id,metadata_json FROM retrieval_chunks WHERE record_id=? ORDER BY chunk_index",
            (review_id,),
        ).fetchall()
        assert len(chunks) >= 3
        assert all(json.loads(row["metadata_json"])["watched_date"] == "2020-01-02" for row in chunks)
        vectors = conn.execute("SELECT vector,vector_dimension FROM retrieval_embeddings").fetchall()
        for row in vectors:
            vector = np.frombuffer(row["vector"], dtype=np.float32)
            assert row["vector_dimension"] == vector.size
            assert np.isclose(np.linalg.norm(vector), 1.0)

    stable_ids = [row["chunk_id"] for row in chunks]
    second_embedder = FakeEmbedder("v1")
    second = refresh_retrieval_index(str(db), second_embedder)
    assert second["embedded"] == 0
    assert second["reused"] == second["chunks"]
    with mdb.connect(db) as conn:
        same_ids = [row[0] for row in conn.execute(
            "SELECT chunk_id FROM retrieval_chunks WHERE record_id=? ORDER BY chunk_index", (review_id,)
        ).fetchall()]
    assert same_ids == stable_ids

    with mdb.connect(db) as conn:
        conn.execute(
            "UPDATE retrieval_corpus SET text=? WHERE content_type='tmdb_overview'",
            ("An edited plot summary.",),
        )
    changed_embedder = FakeEmbedder("v1")
    changed = refresh_retrieval_index(str(db), changed_embedder)
    assert changed["embedded"] == 1

    new_model = FakeEmbedder("v2", dimensions=5)
    rebuilt = refresh_retrieval_index(str(db), new_model)
    assert rebuilt["embedded"] == rebuilt["chunks"]
    with mdb.connect(db) as conn:
        assert [row[0] for row in conn.execute("SELECT DISTINCT model_revision FROM retrieval_embeddings").fetchall()] == ["v2"]
        assert [row[0] for row in conn.execute("SELECT DISTINCT vector_dimension FROM retrieval_embeddings").fetchall()] == [5]

    with mdb.connect(db) as conn:
        conn.execute("DELETE FROM retrieval_corpus WHERE record_id=?", (review_id,))
    removed = refresh_retrieval_index(str(db), new_model)
    assert removed["chunks"] == 1
    with mdb.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM retrieval_chunks").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM retrieval_embeddings").fetchone()[0] == 1


def test_refresh_handles_empty_corpus_without_loading_vectors(tmp_path):
    db = tmp_path / "empty.sqlite"
    result = refresh_retrieval_index(str(db), FakeEmbedder())
    assert result["source_records"] == 0
    assert result["chunks"] == 0
    assert result["embeddings"] == 0
