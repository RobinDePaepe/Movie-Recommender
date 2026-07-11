"""Tests for the Curated Weeks builder."""
from __future__ import annotations

import pandas as pd
import pytest

from curator import (
    BASE_ROLES,
    _overlap,
    _role_sequence,
    anchor_options,
    build_curated_list,
)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame([
        {"Name": "The Anchor", "Year": 2000, "genres": ["Thriller", "Crime"],
         "directors": ["Ann Auteur"], "cast": ["Star A"], "keywords": ["heist", "vault", "loyalty"],
         "overview": "A crew plans an intricate vault heist."},
        {"Name": "Director Reunion", "Year": 2004, "genres": ["Drama"],
         "directors": ["Ann Auteur"], "cast": ["Star B"], "keywords": ["family", "loss"],
         "overview": "The same director explores grief."},
        {"Name": "Theme Cousin", "Year": 2011, "genres": ["Thriller"],
         "directors": ["Bob Other"], "cast": ["Star C"], "keywords": ["heist", "vault", "betrayal"],
         "overview": "Another intricate vault heist with betrayal."},
        {"Name": "Unrelated Rom", "Year": 2015, "genres": ["Romance"],
         "directors": ["Cara Third"], "cast": ["Star D"], "keywords": ["wedding", "paris"],
         "overview": "A gentle love story in Paris."},
        {"Name": "Second Auteur", "Year": 2008, "genres": ["Thriller"],
         "directors": ["Ann Auteur"], "cast": ["Star E"], "keywords": ["chase"],
         "overview": "The director's chase thriller."},
    ])


def _data() -> dict:
    # Everything is "watchlist" so the pool is unwatched by default.
    meta = _metadata()
    watchlist = meta[["Name", "Year"]].copy()
    watchlist["movie_id"] = watchlist.apply(lambda r: f"{r['Name'].lower()} ({int(r['Year'])})", axis=1)
    empty = pd.DataFrame(columns=["Name", "Year", "movie_id"])
    return {
        "watchlist": watchlist,
        "watched": empty.copy(),
        "ratings": empty.copy(),
        "likes": empty.copy(),
        "diary": empty.copy(),
        "lists": pd.DataFrame(),
    }


# --- role sequence -----------------------------------------------------------

def test_role_sequence_length_matches_request():
    assert len(_role_sequence(5)) == 5
    assert len(_role_sequence(10)) == 10


def test_role_sequence_includes_or_excludes_anchor():
    assert "Anchor movie" in _role_sequence(7, include_anchor=True)
    assert "Anchor movie" not in _role_sequence(7, include_anchor=False)


def test_role_sequence_single_movie():
    assert _role_sequence(1, include_anchor=True) == ["Anchor movie"]
    assert _role_sequence(1, include_anchor=False) == ["Companion film"]


def test_overlap_counts_shared_items():
    n, matches = _overlap(["a", "b", "c"], ["b", "c", "d"])
    assert n == 2
    assert matches == ["b", "c"]


# --- build_curated_list ------------------------------------------------------

def test_build_curated_list_returns_ordered_days_with_anchor():
    result = build_curated_list("the anchor (2000)", _data(), _metadata(), total_movies=4, include_anchor=True)
    assert not result.empty
    assert result["day"].tolist() == sorted(result["day"].tolist())
    assert "the anchor (2000)" in set(result["movie_id"])
    assert (result["role"] == "Anchor movie").sum() == 1


def test_build_curated_list_excludes_anchor_when_requested():
    result = build_curated_list("the anchor (2000)", _data(), _metadata(), total_movies=4, include_anchor=False)
    assert "the anchor (2000)" not in set(result["movie_id"])


def test_build_curated_list_unknown_anchor_raises():
    with pytest.raises(ValueError):
        build_curated_list("does not exist (1900)", _data(), _metadata())


def test_build_curated_list_respects_director_cap_when_not_director_focused():
    # Three films share "Ann Auteur"; the Balanced style caps at max_per_director (2).
    result = build_curated_list("the anchor (2000)", _data(), _metadata(),
                                total_movies=6, style="Balanced", include_anchor=True)
    ann_films = [d for d in result["directors"] if isinstance(d, list) and "Ann Auteur" in d]
    # Anchor (Ann) + at most 2 more Ann films — soft cap keeps it from flooding.
    assert len(ann_films) <= 3


def test_build_curated_list_director_focused_allows_more_same_director():
    balanced = build_curated_list("the anchor (2000)", _data(), _metadata(),
                                  total_movies=5, style="Balanced", include_anchor=True)
    directed = build_curated_list("the anchor (2000)", _data(), _metadata(),
                                  total_movies=5, style="Director-focused", include_anchor=True)

    def ann_count(df):
        return sum(1 for d in df["directors"] if isinstance(d, list) and "Ann Auteur" in d)

    assert ann_count(directed) >= ann_count(balanced)


# --- anchor_options ----------------------------------------------------------

def test_anchor_options_labels_and_sources():
    data = _data()
    opts = anchor_options(_metadata(), data)
    assert not opts.empty
    assert "label" in opts.columns
    # All films are watchlisted, so each should carry a Watchlist source.
    assert all("Watchlist" in s for s in opts["source_labels"])


def test_anchor_options_empty_metadata():
    assert anchor_options(pd.DataFrame()).empty
