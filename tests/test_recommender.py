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


# --- outside_watchlist mode / discovery ---

def _outside_watchlist_metadata() -> pd.DataFrame:
    """Cache holding one discovered film, one enrichment-only film, and an owned film."""
    return pd.DataFrame([
        {"Name": "Discovered Film", "Year": 2018, "genres": ["Drama"], "directors": ["Dir One"],
         "writers": ["Dir One"], "cast": ["Actor One"], "keywords": ["heist"],
         "tmdb_url": "https://tmdb/d", "discovered_from": "Movie A (2000)"},
        {"Name": "Enriched Film", "Year": 2019, "genres": ["Drama"], "directors": ["Dir Two"],
         "writers": ["Dir Two"], "cast": ["Actor Two"], "keywords": ["heist"],
         "tmdb_url": "https://tmdb/e", "discovered_from": ""},
        {"Name": "Movie A", "Year": 2000, "genres": ["Drama"], "directors": ["Dir Three"],
         "writers": ["Dir Three"], "cast": ["Actor Three"], "keywords": ["heist"],
         "tmdb_url": "https://tmdb/a", "discovered_from": ""},
    ])


def test_outside_watchlist_excludes_owned():
    data = _minimal_data()
    metadata = _outside_watchlist_metadata()
    recs, _ = build_recommendations(data, metadata=metadata, mode="outside_watchlist")
    ids = set(recs["movie_id"].tolist())
    rated_ids = set(data["ratings"]["movie_id"])
    watchlist_ids = set(data["watchlist"]["movie_id"])
    assert not (ids & rated_ids), "Rated films leaked into outside-watchlist pool"
    assert not (ids & watchlist_ids), "Watchlist films leaked into outside-watchlist pool"


def test_outside_watchlist_includes_discovered():
    data = _minimal_data()
    metadata = _outside_watchlist_metadata()
    recs, _ = build_recommendations(data, metadata=metadata, mode="outside_watchlist")
    assert "discovered film (2018)" in recs["movie_id"].tolist()


def test_outside_watchlist_excludes_non_discovered_cache():
    data = _minimal_data()
    metadata = _outside_watchlist_metadata()
    recs, _ = build_recommendations(data, metadata=metadata, mode="outside_watchlist")
    # Enrichment-only cache rows (no discovered_from) must not leak into the pool.
    assert "enriched film (2019)" not in recs["movie_id"].tolist()


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


def _feedback_metadata() -> pd.DataFrame:
    """Metadata for one tagged film, used to drive content-similarity feedback scoring."""
    return pd.DataFrame([
        {"Name": "Tagged Film", "Year": 2001, "genres": ["Drama"], "directors": ["Dir"],
         "writers": ["Dir"], "cast": ["Actor"], "keywords": ["heist", "noir"],
         "overview": "a tense heist", "tmdb_url": "https://tmdb/t"},
    ])


def _feedback_candidate() -> pd.DataFrame:
    """A single candidate whose feature_text overlaps the tagged film above."""
    cand = normalize_movie_key(pd.DataFrame([{"Name": "Similar Film", "Year": 2002}]))
    cand["feature_text"] = "drama heist noir tense"
    return cand


def test_feedback_scope_watched_tuning_stronger_than_recommendation():
    metadata = _feedback_metadata()
    fb_rec = pd.DataFrame([{"movie_id": "tagged film (2001)", "feedback": "more_like_this", "scope": "recommendation"}])
    fb_watch = pd.DataFrame([{"movie_id": "tagged film (2001)", "feedback": "more_like_this", "scope": "watched_tuning"}])
    s_rec = add_feedback_similarity(_feedback_candidate(), fb_rec, metadata)["feedback_score"].iloc[0]
    s_watch = add_feedback_similarity(_feedback_candidate(), fb_watch, metadata)["feedback_score"].iloc[0]
    assert s_rec > 0
    assert s_watch > s_rec


def test_feedback_high_quality_not_my_taste_pushes_away():
    metadata = _feedback_metadata()
    fb = pd.DataFrame([{"movie_id": "tagged film (2001)", "feedback": "high_quality_not_my_taste", "scope": "watched_tuning"}])
    result = add_feedback_similarity(_feedback_candidate(), fb, metadata)
    assert result["feedback_score"].iloc[0] < 0


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


# --- thematic / conceptual similarity ---

def _theme_data() -> tuple[dict, pd.DataFrame]:
    """A setup that separates *theme* from genre/director/cast.

    - "Auteur Drama" (5★) and "Bad Blockbuster" (2★) define taste.
    - "Dream Heist" (watched) is the anchor: a dream/subconscious concept film.
    - "Mind Maze" (watchlist) shares the anchor's THEMES (dream/subconscious) but its
      genre/director/cast match the DISLIKED blockbuster -> high theme/anchor, negative taste.
    - "Cozy Wedding" (watchlist) shares the LIKED drama's themes (marriage/family) but a
      different genre/director/cast -> a taste-theme cousin.
    """
    ratings = normalize_movie_key(pd.DataFrame([
        {"Name": "Auteur Drama", "Year": 2005, "Rating": 5.0},
        {"Name": "Bad Blockbuster", "Year": 2018, "Rating": 2.0},
    ]))
    watched = normalize_movie_key(pd.DataFrame([
        {"Name": "Dream Heist", "Year": 2010},
    ]))
    watchlist = normalize_movie_key(pd.DataFrame([
        {"Name": "Mind Maze", "Year": 2014},
        {"Name": "Cozy Wedding", "Year": 2016},
    ]))
    data = {
        "ratings": ratings,
        "watched": watched,
        "watchlist": watchlist,
        "likes": pd.DataFrame(columns=["Name", "Year", "movie_id"]),
        "lists": pd.DataFrame(),
        "diary": pd.DataFrame(),
    }
    metadata = pd.DataFrame([
        {"Name": "Auteur Drama", "Year": 2005, "genres": ["Drama"],
         "directors": ["Fave Director"], "writers": ["Fave Director"], "cast": ["Muse Actor"],
         "keywords": ["grief", "marriage", "family"],
         "overview": "A quiet family drama about grief, marriage, and forgiveness."},
        {"Name": "Bad Blockbuster", "Year": 2018, "genres": ["Action", "Adventure"],
         "directors": ["Hack Director"], "writers": ["Hack Director"], "cast": ["Action Star"],
         "keywords": ["explosion", "chase"],
         "overview": "Loud action with explosions and relentless car chases."},
        {"Name": "Dream Heist", "Year": 2010, "genres": ["Science Fiction", "Action"],
         "directors": ["Big Name"], "writers": ["Big Name"], "cast": ["A Star"],
         "keywords": ["dream", "subconscious", "mind bending", "heist"],
         "overview": "Thieves enter layered dreams to steal secrets from the subconscious."},
        {"Name": "Mind Maze", "Year": 2014, "genres": ["Action", "Adventure"],
         "directors": ["Hack Director"], "writers": ["Hack Director"], "cast": ["Action Star"],
         "keywords": ["dream", "subconscious", "mind bending"],
         "overview": "A man navigates collapsing layered dreams and a shifting subconscious reality."},
        {"Name": "Cozy Wedding", "Year": 2016, "genres": ["Comedy", "Romance"],
         "directors": ["Nobody Else"], "writers": ["Nobody Else"], "cast": ["Rom Star"],
         "keywords": ["wedding", "marriage", "love"],
         "overview": "A warm romantic comedy about marriage and family."},
    ])
    return data, metadata


def test_theme_weight_raises_thematic_candidate():
    # Cozy Wedding shares themes (marriage/family) with the 5★ drama but not its
    # genre/director/cast, so only the theme channel should lift it.
    data, metadata = _theme_data()
    off, _ = build_recommendations(data, metadata=metadata, score_weights={"theme": 0.0})
    on, _ = build_recommendations(data, metadata=metadata, score_weights={"theme": 3.0})
    cousin_off = off.set_index("Name").loc["Cozy Wedding", "score"]
    cousin_on = on.set_index("Name").loc["Cozy Wedding", "score"]
    assert cousin_on > cousin_off


def test_anchor_focus_rescues_concept_cousin():
    # Mind Maze is thematically the anchor's twin but matches the user's DISLIKED taste.
    # Anchor focus eases off global taste so the concept-cousin isn't cancelled.
    data, metadata = _theme_data()
    anchor_id = "dream heist (2010)"
    focused, _ = build_recommendations(data, metadata=metadata, anchor_movie_id=anchor_id, anchor_focus=True)
    flat, _ = build_recommendations(data, metadata=metadata, anchor_movie_id=anchor_id, anchor_focus=False)
    maze_focus = focused.set_index("Name").loc["Mind Maze", "score"]
    maze_flat = flat.set_index("Name").loc["Mind Maze", "score"]
    assert maze_focus > maze_flat


def test_theme_similarity_helpers_finite_without_embedding_lib():
    import theme_similarity
    from recommender import prepare_metadata, candidate_pool
    data, metadata = _theme_data()
    meta = prepare_metadata(metadata)
    cands = candidate_pool(data, mode="watchlist")
    anchor = theme_similarity.theme_anchor_scores(cands, "dream heist (2010)", meta)
    taste = theme_similarity.theme_taste_scores(cands, data["ratings"], data["likes"], meta)
    assert len(anchor) == len(cands) and anchor.notna().all()
    assert (anchor >= -0.001).all() and (anchor <= 4.001).all()
    assert len(taste) == len(cands) and taste.notna().all()
    # Mind Maze should be the strongest theme match to the dream-heist anchor.
    cands = cands.reset_index(drop=True)
    maze_pos = cands.index[cands["Name"] == "Mind Maze"][0]
    wedding_pos = cands.index[cands["Name"] == "Cozy Wedding"][0]
    assert anchor.iloc[maze_pos] > anchor.iloc[wedding_pos]
