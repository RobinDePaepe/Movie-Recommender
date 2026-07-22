"""Reflection panel: category sub-ratings for one film, plus facet-level evidence to inform them.

Evidence (director/cast/theme history) is a lookup against the existing signal machinery, not a
new computation. `suggested_overall_rating` combines your own category ratings into one number;
persistence lives in `movie_database.save_reflection`/`load_latest_reflection`.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import theme_similarity
from movie_database import movie_id as build_movie_id
from recommender import _as_list, prepare_metadata

REFLECTION_CATEGORIES = [
    "Story & Writing",
    "Direction",
    "Acting",
    "Visuals & Sound",
    "Emotional Impact",
    "Rewatchability",
]


def _split_title_year(query: str) -> Tuple[str, Optional[int]]:
    match = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", query.strip())
    if match:
        return match.group(1).strip(), int(match.group(2))
    return query.strip(), None


def resolve_for_reflection(
    query: str,
    data: Dict[str, pd.DataFrame],
    metadata: pd.DataFrame | None,
    tmdb_client=None,
) -> Dict[str, Any]:
    """Resolve a movie_id or free-text title into one consistent shape.

    Handles three cases: in DB + rated, in DB + watchlisted (rating=None), or not in DB
    (TMDb search/fetch, nothing written anywhere).
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Enter a movie title or movie_id.")

    data = data or {}
    meta = prepare_metadata(metadata) if metadata is not None and not metadata.empty else pd.DataFrame()

    mid = query.lower()
    title_guess, year_guess = _split_title_year(query)

    known_ids: set = set()
    for key in ("ratings", "watchlist", "watched", "likes"):
        frame = data.get(key, pd.DataFrame())
        if frame is not None and not frame.empty and "movie_id" in frame.columns:
            known_ids |= set(frame["movie_id"].dropna())
    meta_ids = set(meta["movie_id"]) if not meta.empty else set()
    in_db = mid in known_ids or mid in meta_ids

    name, year = title_guess, year_guess
    ratings = data.get("ratings", pd.DataFrame())
    for key in ("ratings", "watchlist", "watched", "likes"):
        frame = data.get(key, pd.DataFrame())
        if frame is not None and not frame.empty and "movie_id" in frame.columns:
            match = frame[frame["movie_id"] == mid]
            if not match.empty:
                name, year = match.iloc[0]["Name"], match.iloc[0]["Year"]
                break

    meta_row: Optional[Dict[str, Any]] = None
    if mid in meta_ids:
        meta_row = meta.loc[meta["movie_id"] == mid].iloc[0].to_dict()

    if meta_row is None and in_db and tmdb_client is not None:
        fetched = tmdb_client.fetch_movie_metadata(name, year)
        if fetched.get("tmdb_found"):
            meta_row = fetched

    if meta_row is None and not in_db:
        if tmdb_client is None:
            raise ValueError(f'"{query}" isn\'t in your library, and no TMDb API key is set to look it up.')
        fetched = tmdb_client.fetch_movie_metadata(title_guess, year_guess)
        if not fetched.get("tmdb_found"):
            raise ValueError(f'No TMDb match found for "{query}".')
        meta_row = fetched
        name = fetched.get("name") or title_guess
        year = fetched.get("year") or year_guess
        mid = build_movie_id(name, year)

    if meta_row is None:
        meta_row = {}

    rating = None
    if ratings is not None and not ratings.empty and "movie_id" in ratings.columns:
        rmatch = ratings[ratings["movie_id"] == mid]
        if not rmatch.empty:
            rv = pd.to_numeric(rmatch.iloc[0].get("Rating"), errors="coerce")
            rating = float(rv) if pd.notna(rv) else None

    return {
        "movie_id": mid,
        "title": name,
        "year": year,
        "genres": _as_list(meta_row.get("genres", [])),
        "director": _as_list(meta_row.get("directors", [])),
        "cast": _as_list(meta_row.get("cast", [])),
        "keywords": _as_list(meta_row.get("keywords", [])),
        "overview": str(meta_row.get("overview", "") or ""),
        "poster_url": meta_row.get("poster_url", "") or "",
        "in_db": in_db,
        "rating": rating,
    }


def _entity_history(
    name: str, role_col: str, meta: pd.DataFrame, ratings: pd.DataFrame, exclude_movie_id: str | None = None
) -> List[Tuple[str, int, float]]:
    if meta is None or meta.empty or ratings is None or ratings.empty or role_col not in meta.columns:
        return []
    mask = meta[role_col].apply(lambda lst: isinstance(lst, list) and name in lst)
    sub = meta.loc[mask, ["movie_id"]]
    if exclude_movie_id is not None:
        sub = sub[sub["movie_id"] != exclude_movie_id]
    if sub.empty:
        return []
    r = ratings.copy()
    r["Rating"] = pd.to_numeric(r.get("Rating"), errors="coerce")
    r = r.dropna(subset=["Rating"])
    merged = sub.merge(r[["movie_id", "Name", "Year", "Rating"]], on="movie_id", how="inner")
    if merged.empty:
        return []
    merged = merged.sort_values("Rating", ascending=False)
    return [(row["Name"], row["Year"], float(row["Rating"])) for _, row in merged.iterrows()]


def _richest_entity_facet(
    names: List[str], role_col: str, meta: pd.DataFrame, ratings: pd.DataFrame, facet_label: str, exclude_movie_id: str | None = None
) -> Dict[str, Any]:
    names = [n for n in (names or []) if n]
    best_name, best_history = None, []
    for name in names:
        history = _entity_history(name, role_col, meta, ratings, exclude_movie_id=exclude_movie_id)
        if len(history) > len(best_history):
            best_name, best_history = name, history
    if best_name is None and names:
        best_name = names[0]
    return {"facet": facet_label, "value": best_name, "history": best_history, "n": len(best_history)}


def director_evidence(resolved: Dict[str, Any], metadata: pd.DataFrame | None, ratings: pd.DataFrame) -> Dict[str, Any]:
    meta = prepare_metadata(metadata) if metadata is not None and not metadata.empty else pd.DataFrame()
    return _richest_entity_facet(resolved.get("director"), "directors", meta, ratings, "director", exclude_movie_id=resolved.get("movie_id"))


def cast_evidence(resolved: Dict[str, Any], metadata: pd.DataFrame | None, ratings: pd.DataFrame) -> Dict[str, Any]:
    meta = prepare_metadata(metadata) if metadata is not None and not metadata.empty else pd.DataFrame()
    return _richest_entity_facet(resolved.get("cast"), "cast", meta, ratings, "cast", exclude_movie_id=resolved.get("movie_id"))


def theme_evidence(resolved: Dict[str, Any], data: Dict[str, pd.DataFrame], metadata: pd.DataFrame | None, k: int = 5) -> Dict[str, Any]:
    meta = prepare_metadata(metadata) if metadata is not None and not metadata.empty else pd.DataFrame()
    ratings = (data or {}).get("ratings", pd.DataFrame())
    query_id = resolved["movie_id"]

    if meta.empty or query_id not in set(meta["movie_id"]):
        extra = pd.DataFrame([{"movie_id": query_id, "keywords": resolved.get("keywords", []), "overview": resolved.get("overview", "")}])
        meta_for_theme = pd.concat([meta, extra], ignore_index=True) if not meta.empty else extra
    else:
        meta_for_theme = meta

    if ratings is None or ratings.empty:
        return {"facet": "theme", "value": "thematic profile", "history": [], "n": 0}

    r = ratings.copy()
    r["Rating"] = pd.to_numeric(r.get("Rating"), errors="coerce")
    r = r.dropna(subset=["Rating"])
    r = r[r["movie_id"] != query_id]
    candidate_ids = r["movie_id"].dropna().unique().tolist()
    if not candidate_ids:
        return {"facet": "theme", "value": "thematic profile", "history": [], "n": 0}

    pairs = theme_similarity.most_similar_by_theme(query_id, meta_for_theme, candidate_ids=candidate_ids, k=k)
    rating_lookup = r.set_index("movie_id")
    history = []
    for mid, _sim in pairs:
        if mid in rating_lookup.index:
            row = rating_lookup.loc[mid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            history.append((row["Name"], row["Year"], float(row["Rating"])))
    return {"facet": "theme", "value": "thematic profile", "history": history, "n": len(history)}


def suggested_overall_rating(category_ratings: Dict[str, float], weights: Dict[str, float] | None = None) -> float:
    """Weighted average of category ratings, on the same 0.5-5.0 scale as the inputs.

    Categories missing from `weights` default to a weight of 1.0. A category with a zero (or
    missing) weight contributes nothing to the result. Returns 0.0 if there's nothing to average.
    """
    weights = weights or {}
    total_weight = 0.0
    weighted_sum = 0.0
    for category, rating in category_ratings.items():
        w = weights.get(category, 1.0)
        weighted_sum += float(rating) * w
        total_weight += w
    if total_weight <= 0:
        return 0.0
    return weighted_sum / total_weight


def build_facets(resolved: Dict[str, Any], data: Dict[str, pd.DataFrame], metadata: pd.DataFrame | None) -> List[Dict[str, Any]]:
    ratings = (data or {}).get("ratings", pd.DataFrame())
    return [
        director_evidence(resolved, metadata, ratings),
        cast_evidence(resolved, metadata, ratings),
        theme_evidence(resolved, data, metadata),
    ]
