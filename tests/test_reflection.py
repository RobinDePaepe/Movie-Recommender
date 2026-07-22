"""Tests for the reflection panel: suggested-rating math and DB persistence round-trip."""
from __future__ import annotations

from pathlib import Path

import pytest

from movie_database import load_latest_reflection, save_reflection
from reflection import REFLECTION_CATEGORIES, suggested_overall_rating


def test_suggested_overall_rating_equal_weights():
    ratings = {"Story & Writing": 4.0, "Direction": 5.0, "Acting": 3.0}
    assert suggested_overall_rating(ratings) == pytest.approx(4.0)


def test_suggested_overall_rating_skewed_weights():
    ratings = {"Story & Writing": 5.0, "Direction": 1.0}
    weights = {"Story & Writing": 3.0, "Direction": 1.0}
    # (5*3 + 1*1) / 4 = 4.0
    assert suggested_overall_rating(ratings, weights) == pytest.approx(4.0)


def test_suggested_overall_rating_missing_weight_defaults_to_one():
    ratings = {"Story & Writing": 4.0, "Direction": 2.0}
    weights = {"Story & Writing": 2.0}  # Direction defaults to 1.0
    # (4*2 + 2*1) / 3 = 3.333...
    assert suggested_overall_rating(ratings, weights) == pytest.approx(10 / 3)


def test_suggested_overall_rating_zero_total_weight_returns_zero():
    ratings = {"Story & Writing": 4.0}
    weights = {"Story & Writing": 0.0}
    assert suggested_overall_rating(ratings, weights) == 0.0


def test_suggested_overall_rating_empty_input():
    assert suggested_overall_rating({}) == 0.0


def test_save_and_load_reflection_round_trip(tmp_path: Path):
    db_path = tmp_path / "test.sqlite"
    category_payload = {
        "Story & Writing": (4.5, "Tight plotting."),
        "Direction": (5.0, "Confident, patient camera work."),
    }
    for category in REFLECTION_CATEGORIES:
        category_payload.setdefault(category, (3.0, ""))

    save_reflection(
        "reflect test film (2020)", "Reflect Test Film", 2020,
        category_payload, overall_rating=4.5, overall_note="A near-favorite.",
        db_path=db_path,
    )

    loaded = load_latest_reflection("reflect test film (2020)", db_path=db_path)
    assert loaded["categories"]["Story & Writing"]["rating"] == 4.5
    assert loaded["categories"]["Story & Writing"]["note"] == "Tight plotting."
    assert loaded["overall"]["rating"] == 4.5
    assert loaded["overall"]["note"] == "A near-favorite."

    import movie_database
    with movie_database.connect(db_path) as conn:
        row = conn.execute(
            "SELECT rating, source FROM ratings WHERE movie_id=?", ("reflect test film (2020)",)
        ).fetchone()
    assert row["rating"] == 4.5
    assert row["source"] == "reflection"


def test_save_reflection_updates_official_rating_and_logs_history(tmp_path: Path):
    db_path = tmp_path / "test.sqlite"
    payload = {cat: (3.0, "") for cat in REFLECTION_CATEGORIES}

    save_reflection("history film (2019)", "History Film", 2019, payload, overall_rating=3.0, db_path=db_path)
    save_reflection("history film (2019)", "History Film", 2019, payload, overall_rating=4.0, db_path=db_path)

    import movie_database
    with movie_database.connect(db_path) as conn:
        history_rows = conn.execute(
            "SELECT old_rating, new_rating, source FROM rating_history WHERE movie_id=?",
            ("history film (2019)",),
        ).fetchall()
        current = conn.execute(
            "SELECT rating FROM ratings WHERE movie_id=?", ("history film (2019)",)
        ).fetchone()

    assert current["rating"] == 4.0
    assert len(history_rows) == 1
    assert history_rows[0]["old_rating"] == 3.0
    assert history_rows[0]["new_rating"] == 4.0
    assert history_rows[0]["source"] == "reflection"


def test_load_latest_reflection_returns_most_recent_per_category(tmp_path: Path):
    db_path = tmp_path / "test.sqlite"
    payload = {cat: (2.0, "first pass") for cat in REFLECTION_CATEGORIES}
    save_reflection("rewatch film (2018)", "Rewatch Film", 2018, payload, overall_rating=2.0, db_path=db_path)

    payload2 = {cat: (4.5, "after rewatch, loved it more") for cat in REFLECTION_CATEGORIES}
    save_reflection("rewatch film (2018)", "Rewatch Film", 2018, payload2, overall_rating=4.5, db_path=db_path)

    loaded = load_latest_reflection("rewatch film (2018)", db_path=db_path)
    for category in REFLECTION_CATEGORIES:
        assert loaded["categories"][category]["rating"] == 4.5
        assert loaded["categories"][category]["note"] == "after rewatch, loved it more"
    assert loaded["overall"]["rating"] == 4.5
