"""Smoke tests for the recommender engine."""
from __future__ import annotations

import pandas as pd
import pytest

import datetime

from recommender import (
    FEEDBACK_LABELS,
    _as_list,
    add_feedback_similarity,
    add_heuristic_scores,
    build_recommendations,
    candidate_pool,
    normalize_movie_key,
)


def _metadata_data() -> tuple[dict, pd.DataFrame]:
    """Data + metadata where one watchlist film mirrors a 5★ film and another is just recent."""
    current_year = datetime.date.today().year
    ratings = normalize_movie_key(pd.DataFrame([
        {"Name": "Liked Film", "Year": 2005, "Rating": 5.0},
    ]))
    watchlist = normalize_movie_key(pd.DataFrame([
        {"Name": "Twin Film", "Year": 2006},
        {"Name": "Fresh Film", "Year": current_year},
    ]))
    data = {
        "ratings": ratings,
        "watched": pd.DataFrame(columns=["Name", "Year", "movie_id"]),
        "watchlist": watchlist,
        "likes": pd.DataFrame(columns=["Name", "Year", "movie_id"]),
        "lists": pd.DataFrame(),
        "diary": pd.DataFrame(),
    }
    metadata = pd.DataFrame([
        {"Name": "Liked Film", "Year": 2005, "genres": ["Drama", "Crime"],
         "directors": ["Jane Director"], "writers": ["Jane Director"], "cast": ["Star One"], "keywords": ["heist", "noir"]},
        {"Name": "Twin Film", "Year": 2006, "genres": ["Drama", "Crime"],
         "directors": ["Jane Director"], "writers": ["Jane Director"], "cast": ["Star One"], "keywords": ["heist", "noir"]},
        {"Name": "Fresh Film", "Year": current_year, "genres": ["Comedy", "Family"],
         "directors": ["Other Person"], "writers": ["Other Person"], "cast": ["Star Two"], "keywords": ["wedding"]},
    ])
    return data, metadata


def _minimal_data() -> dict:
    ratings = normalize_movie_key(pd.DataFrame([
        {"Name": "Movie A", "Year": 2000, "Rating": 5.0},
        {"Name": "Movie B", "Year": 2010, "Rating": 4.0},
        {"Name": "Movie C", "Year": 2015, "Rating": 3.0},
    ]))
    watchlist = normalize_movie_key(pd.DataFrame([
        {"Name": "Movie D", "Year": 2020},
        {"Name": "Movie E", "Year": 2022},
    ]))
    return {
        "ratings": ratings,
        "watched": pd.DataFrame(columns=["Name", "Year", "movie_id"]),
        "watchlist": watchlist,
        "likes": pd.DataFrame(columns=["Name", "Year", "movie_id"]),
        "lists": pd.DataFrame(),
        "diary": pd.DataFrame(),
    }


# --- _as_list ---

def test_as_list_comma_string():
    assert _as_list("a, b, c") == ["a", "b", "c"]


def test_as_list_json_array():
    assert _as_list('["Drama", "Comedy"]') == ["Drama", "Comedy"]


def test_as_list_python_list():
    assert _as_list(["x", "y"]) == ["x", "y"]


def test_as_list_empty_string():
    assert _as_list("") == []


def test_as_list_none():
    assert _as_list(None) == []


# --- normalize_movie_key ---

def test_normalize_movie_key_strips_and_lowercases():
    df = normalize_movie_key(pd.DataFrame([{"Name": " The Movie ", "Year": "1999"}]))
    assert df["movie_id"].iloc[0] == "the movie (1999)"


def test_normalize_movie_key_na_year():
    df = normalize_movie_key(pd.DataFrame([{"Name": "Untitled", "Year": None}]))
    assert "<NA>" in df["movie_id"].iloc[0]


# --- FEEDBACK_LABELS ---

def test_feedback_labels_all_have_weight_and_description():
    for key, val in FEEDBACK_LABELS.items():
        assert "weight" in val, f"Missing weight for {key}"
        assert "description" in val, f"Missing description for {key}"


def test_feedback_labels_positive_and_negative_exist():
    weights = [v["weight"] for v in FEEDBACK_LABELS.values()]
    assert any(w > 0 for w in weights), "No positive feedback labels"
    assert any(w < 0 for w in weights), "No negative feedback labels"


# --- candidate_pool ---

def test_candidate_pool_watchlist_excludes_watched():
    data = _minimal_data()
    # Mark Movie D as watched so it should be excluded.
    data["watched"] = normalize_movie_key(pd.DataFrame([{"Name": "Movie D", "Year": 2020}]))
    pool = candidate_pool(data, mode="watchlist")
    assert "movie d (2020)" not in pool["movie_id"].tolist()


def test_candidate_pool_watchlist_not_empty():
    data = _minimal_data()
    pool = candidate_pool(data, mode="watchlist")
    assert not pool.empty


# --- build_recommendations ---

def test_build_recommendations_no_metadata_returns_results():
    data = _minimal_data()
    recs, decade_prefs = build_recommendations(data, metadata=None)
    assert not recs.empty
    assert "score" in recs.columns
    assert "why" in recs.columns


def test_build_recommendations_sorted_descending():
    data = _minimal_data()
    recs, _ = build_recommendations(data)
    scores = recs["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_build_recommendations_required_columns():
    data = _minimal_data()
    recs, _ = build_recommendations(data)
    for col in ["Name", "Year", "movie_id", "score", "why", "why_details"]:
        assert col in recs.columns, f"Missing column: {col}"


# --- add_feedback_similarity ---

def test_add_feedback_similarity_empty_feedback_returns_zero():
    data = _minimal_data()
    pool = candidate_pool(data)
    from recommender import add_heuristic_scores, add_content_similarity
    pool, _ = add_heuristic_scores(pool, data)
    pool, _ = add_content_similarity(pool, data["ratings"], data["likes"], pd.DataFrame())
    result = add_feedback_similarity(pool, pd.DataFrame(), None)
    assert "feedback_score" in result.columns
    assert (result["feedback_score"] == 0.0).all()


def test_add_feedback_similarity_positive_label_increases_score():
    data = _minimal_data()
    pool = candidate_pool(data)
    pool, _ = add_heuristic_scores(pool, data)
    pool["feature_text"] = "drama romance emotional"
    feedback = pd.DataFrame([{"movie_id": "movie a (2000)", "feedback": "more_like_this"}])
    result = add_feedback_similarity(pool, feedback, None)
    assert "feedback_score" in result.columns


# --- recency gradient (#2) ---

def test_recency_bonus_decays_with_age():
    current_year = datetime.date.today().year
    data = _minimal_data()
    candidates = normalize_movie_key(pd.DataFrame([
        {"Name": "Now Film", "Year": current_year},
        {"Name": "Five Film", "Year": current_year - 5},
        {"Name": "Old Film", "Year": current_year - 20},
    ]))
    scored, _ = add_heuristic_scores(candidates, data)
    bonus = scored.set_index("Name")["recency_bonus"]
    assert bonus["Now Film"] > bonus["Five Film"] > 0.0
    assert bonus["Old Film"] == 0.0  # beyond the 15-year window


# --- decade shrinkage (#3) ---

def test_decade_score_is_shrunk_toward_mean():
    # One 5★ film in the 1970s plus lower-rated films elsewhere; the sparse high decade
    # should be pulled toward the global mean rather than scoring the raw deviation.
    ratings = normalize_movie_key(pd.DataFrame([
        {"Name": "Lone Classic", "Year": 1975, "Rating": 5.0},
        {"Name": "Modern A", "Year": 2010, "Rating": 3.0},
        {"Name": "Modern B", "Year": 2012, "Rating": 3.0},
        {"Name": "Modern C", "Year": 2014, "Rating": 3.0},
    ]))
    data = {
        "ratings": ratings,
        "watched": pd.DataFrame(columns=["Name", "Year", "movie_id"]),
        "watchlist": normalize_movie_key(pd.DataFrame([{"Name": "Cand", "Year": 1976}])),
        "likes": pd.DataFrame(columns=["Name", "Year", "movie_id"]),
        "lists": pd.DataFrame(),
        "diary": pd.DataFrame(),
    }
    _, decade_pref = add_heuristic_scores(data["watchlist"].copy(), data)
    row = decade_pref[decade_pref["decade"] == "1970s"].iloc[0]
    global_mean = ratings["Rating"].mean()
    raw_score = (row["avg_user_rating"] - global_mean) * 1.2
    assert 0.0 < row["decade_score"] < raw_score


# --- score composition: content beats heuristic-only (#1) ---

def test_content_match_outranks_recent_only_film():
    data, metadata = _metadata_data()
    recs, _ = build_recommendations(data, metadata=metadata)
    scores = recs.set_index("Name")["score"]
    assert scores["Twin Film"] > scores["Fresh Film"]


# --- anchor weighting (#4) ---

def test_anchor_weight_raises_similar_candidate():
    data, metadata = _metadata_data()
    anchor_id = "liked film (2005)"  # Twin Film mirrors this anchor
    off, _ = build_recommendations(data, metadata=metadata, anchor_movie_id=anchor_id,
                                   score_weights={"anchor": 0.0})
    on, _ = build_recommendations(data, metadata=metadata, anchor_movie_id=anchor_id,
                                  score_weights={"anchor": 3.0})
    twin_off = off.set_index("Name").loc["Twin Film", "score"]
    twin_on = on.set_index("Name").loc["Twin Film", "score"]
    assert twin_on > twin_off
