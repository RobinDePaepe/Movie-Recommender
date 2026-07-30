"""Tests for the shared thematic-similarity engine.

These exercise the TF-IDF fallback path (no sentence-transformers required) plus the
scaling and selection logic, so they run without the optional embedding library.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import theme_similarity as ts
from recommender import prepare_metadata


def _meta() -> pd.DataFrame:
    raw = pd.DataFrame([
        {"Name": "Dream Heist", "Year": 2010, "keywords": ["dream", "heist", "subconscious"],
         "overview": "A thief enters dreams to steal secrets."},
        {"Name": "Mind Maze", "Year": 2014, "keywords": ["dream", "subconscious", "memory"],
         "overview": "A woman navigates layered dreams and memory."},
        {"Name": "Cozy Wedding", "Year": 2016, "keywords": ["wedding", "romance"],
         "overview": "A gentle romantic comedy about a countryside wedding."},
    ])
    return prepare_metadata(raw)


def test_theme_text_uses_keywords_and_overview_only():
    row = pd.Series({"keywords": ["dream", "heist"], "overview": "A thief.", "genres": ["Sci-Fi"], "directors": ["X"]})
    text = ts.theme_text(row)
    assert "dream" in text and "thief" in text
    assert "sci-fi" not in text  # genre must not leak into theme text


def test_theme_text_empty_when_no_content():
    assert ts.theme_text(pd.Series({"keywords": [], "overview": ""})) == ""


def test_theme_anchor_scores_ranks_cousin_above_unrelated():
    meta = _meta()
    cands = meta.reset_index(drop=True)
    scores = ts.theme_anchor_scores(cands, "dream heist (2010)", meta)
    assert len(scores) == len(cands)
    maze = cands.index[cands["Name"] == "Mind Maze"][0]
    wedding = cands.index[cands["Name"] == "Cozy Wedding"][0]
    assert scores.iloc[maze] > scores.iloc[wedding]


def test_theme_anchor_scores_bounded():
    meta = _meta()
    scores = ts.theme_anchor_scores(meta.reset_index(drop=True), "dream heist (2010)", meta)
    assert (scores >= -ts.SCORE_SCALE - 0.001).all()
    assert (scores <= ts.SCORE_SCALE + 0.001).all()


def test_theme_anchor_scores_missing_anchor_returns_zeros():
    meta = _meta()
    scores = ts.theme_anchor_scores(meta.reset_index(drop=True), "not here (1900)", meta)
    assert (scores == 0.0).all()


def test_theme_anchor_scores_empty_meta():
    cands = pd.DataFrame({"movie_id": ["a (2000)"]})
    scores = ts.theme_anchor_scores(cands, "a (2000)", pd.DataFrame())
    assert (scores == 0.0).all()


def test_theme_taste_scores_rewards_thematic_match():
    meta = _meta()
    cands = meta.reset_index(drop=True)
    ratings = pd.DataFrame([{"Name": "Dream Heist", "Year": 2010, "Rating": 5.0,
                             "movie_id": "dream heist (2010)"}])
    likes = pd.DataFrame(columns=["movie_id"])
    scores = ts.theme_taste_scores(cands, ratings, likes, meta)
    maze = cands.index[cands["Name"] == "Mind Maze"][0]
    wedding = cands.index[cands["Name"] == "Cozy Wedding"][0]
    assert scores.iloc[maze] > scores.iloc[wedding]


def test_theme_taste_scores_no_positive_ratings_returns_zeros():
    meta = _meta()
    cands = meta.reset_index(drop=True)
    ratings = pd.DataFrame(columns=["Name", "Year", "Rating", "movie_id"])
    likes = pd.DataFrame(columns=["movie_id"])
    scores = ts.theme_taste_scores(cands, ratings, likes, meta)
    assert (scores == 0.0).all()


def test_shared_theme_keywords():
    a = pd.Series({"keywords": ["Dream", "Heist", "Vault"]})
    b = pd.Series({"keywords": ["dream", "heist", "betrayal"]})
    assert ts.shared_theme_keywords(a, b) == ["dream", "heist"]


def test_scale_net_handles_all_zero():
    assert np.array_equal(ts._scale_net(np.zeros(3)), np.zeros(3))
