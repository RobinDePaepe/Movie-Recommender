"""Tests for the SQLite backend: feedback, curated weeks, and load round-trips."""
from __future__ import annotations

import pandas as pd
import pytest

import movie_database as mdb


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "test.sqlite"


# --- helpers -----------------------------------------------------------------

def test_movie_id_key_format():
    assert mdb.movie_id("Inception", 2010) == "inception (2010)"
    assert mdb.movie_id("No Year", None) == "no year (<NA>)"


def test_safe_year_handles_junk():
    assert mdb._safe_year(2010) == 2010
    assert mdb._safe_year(None) is None
    assert mdb._safe_year("not a year") is None


def test_json_serialisation_of_list_columns():
    assert mdb._json(["a", "b"]) == '["a", "b"]'
    assert mdb._json(None) == "[]"
    assert mdb._json("already") == '["already"]'
    assert mdb._json("[\"json\"]") == '["json"]'


# --- feedback ----------------------------------------------------------------

def test_save_and_load_feedback_roundtrip(db):
    mdb.save_feedback_to_db("inception (2010)", "more_like_this", scope="watched_tuning", db_path=db)
    fb = mdb.load_feedback_from_db(db)
    assert len(fb) == 1
    assert fb.iloc[0]["movie_id"] == "inception (2010)"
    assert fb.iloc[0]["feedback"] == "more_like_this"
    assert fb.iloc[0]["scope"] == "watched_tuning"


def test_save_feedback_is_idempotent_per_label(db):
    mdb.save_feedback_to_db("inception (2010)", "more_like_this", db_path=db)
    mdb.save_feedback_to_db("inception (2010)", "more_like_this", db_path=db)
    assert len(mdb.load_feedback_from_db(db)) == 1


def test_remove_feedback(db):
    mdb.save_feedback_to_db("inception (2010)", "more_like_this", db_path=db)
    mdb.save_feedback_to_db("inception (2010)", "rewatchable", db_path=db)
    mdb.remove_feedback_from_db("inception (2010)", ["more_like_this"], db_path=db)
    remaining = mdb.load_feedback_from_db(db)
    assert remaining["feedback"].tolist() == ["rewatchable"]


def test_load_feedback_defaults_scope_to_recommendation(db):
    # Insert a row directly without scope to mimic legacy data.
    mdb.init_db(db)
    with mdb.connect(db) as conn:
        conn.execute("INSERT OR IGNORE INTO movies(movie_id, name, year, created_at, updated_at) VALUES (?, ?, NULL, ?, ?)",
                     ("x (2000)", "x", mdb.utc_now(), mdb.utc_now()))
        conn.execute("INSERT INTO feedback(movie_id, feedback, scope, created_at) VALUES (?, ?, NULL, ?)",
                     ("x (2000)", "more_like_this", mdb.utc_now()))
    fb = mdb.load_feedback_from_db(db)
    assert fb.iloc[0]["scope"] == "recommendation"


# --- movie notes -------------------------------------------------------------

def test_save_and_load_movie_note(db):
    mdb.save_movie_note("inception (2010)", "Loved the layered structure.", db_path=db)
    notes = mdb.load_movie_notes(db)
    assert len(notes) == 1
    assert notes.iloc[0]["movie_id"] == "inception (2010)"
    assert notes.iloc[0]["note"] == "Loved the layered structure."


def test_save_movie_note_upserts(db):
    mdb.save_movie_note("inception (2010)", "first", db_path=db)
    mdb.save_movie_note("inception (2010)", "second", db_path=db)
    notes = mdb.load_movie_notes(db)
    assert len(notes) == 1
    assert notes.iloc[0]["note"] == "second"


def test_empty_note_clears_it(db):
    mdb.save_movie_note("inception (2010)", "something", db_path=db)
    mdb.save_movie_note("inception (2010)", "   ", db_path=db)
    assert mdb.load_movie_notes(db).empty


# --- curated weeks -----------------------------------------------------------

def test_save_and_load_curated_week(db):
    curated = pd.DataFrame([
        {"day": 1, "role": "Anchor movie", "Name": "Inception", "Year": 2010, "movie_id": "inception (2010)"},
        {"day": 2, "role": "Companion film", "Name": "Memento", "Year": 2000, "movie_id": "memento (2000)"},
    ])
    week_id = mdb.save_curated_week("inception (2010)", "Inception", "Balanced", curated, label="Nolan", db_path=db)
    assert isinstance(week_id, int)

    listing = mdb.load_curated_weeks(db)
    assert len(listing) == 1
    assert listing.iloc[0]["anchor_name"] == "Inception"
    assert listing.iloc[0]["total_movies"] == 2
    assert listing.iloc[0]["label"] == "Nolan"

    loaded = mdb.load_curated_week(week_id, db_path=db)
    assert loaded["Name"].tolist() == ["Inception", "Memento"]


def test_load_missing_curated_week_returns_empty(db):
    assert mdb.load_curated_week(999, db_path=db).empty


def test_load_curated_weeks_empty(db):
    mdb.init_db(db)
    assert mdb.load_curated_weeks(db).empty


# --- schema / status ---------------------------------------------------------

def test_init_db_is_idempotent_and_status_reports(db):
    mdb.init_db(db)
    mdb.init_db(db)  # second call must not raise
    status = mdb.database_status(db)
    assert status["exists"] is True
    assert status["feedback"] == 0
    assert status["curated_weeks"] == 0 if "curated_weeks" in status else True


def test_database_status_missing_file(tmp_path):
    assert mdb.database_status(tmp_path / "nope.sqlite") == {"exists": False}
