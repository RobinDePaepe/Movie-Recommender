"""Pure data-shaping helpers for the Analysis page's insight tabs.

Each function is a plain DataFrame-in/DataFrame-out transform with no Streamlit
calls, mirroring the style of recommender.py / theme_similarity.py, so they can
be unit-tested independent of the UI.
"""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from recommender import FEEDBACK_LABELS, compute_entity_affinity
from reflection import REFLECTION_CATEGORIES


def ratings_over_time(ratings: pd.DataFrame, diary: pd.DataFrame) -> pd.DataFrame:
    """Avg rating, diary activity, and rewatch count bucketed by year."""
    rows: Dict[int, Dict[str, float]] = {}

    if not ratings.empty and "Date" in ratings.columns:
        r = ratings.copy()
        r["Rating"] = pd.to_numeric(r.get("Rating"), errors="coerce")
        r["year"] = pd.to_datetime(r["Date"], errors="coerce").dt.year
        r = r.dropna(subset=["year", "Rating"])
        for year, grp in r.groupby(r["year"].astype(int)):
            rows.setdefault(year, {})["avg_rating"] = grp["Rating"].mean()
            rows.setdefault(year, {})["n_ratings"] = len(grp)

    if not diary.empty and "Watched Date" in diary.columns:
        d = diary.copy()
        d["year"] = pd.to_datetime(d["Watched Date"], errors="coerce").dt.year
        d = d.dropna(subset=["year"])
        rewatch_col = pd.to_numeric(d.get("Rewatch"), errors="coerce").fillna(0) if "Rewatch" in d.columns else 0
        d = d.assign(_rewatch=rewatch_col)
        for year, grp in d.groupby(d["year"].astype(int)):
            rows.setdefault(year, {})["n_diary_entries"] = len(grp)
            rows.setdefault(year, {})["n_rewatches"] = int(grp["_rewatch"].sum())

    if not rows:
        return pd.DataFrame(columns=["period", "avg_rating", "n_ratings", "n_diary_entries", "n_rewatches"])

    out = pd.DataFrame.from_dict(rows, orient="index").reset_index().rename(columns={"index": "period"})
    for col in ["avg_rating", "n_ratings", "n_diary_entries", "n_rewatches"]:
        if col not in out.columns:
            out[col] = 0
    out[["n_ratings", "n_diary_entries", "n_rewatches"]] = out[["n_ratings", "n_diary_entries", "n_rewatches"]].fillna(0).astype(int)
    return out.sort_values("period").reset_index(drop=True)


def entity_affinity_table(ratings: pd.DataFrame, metadata: pd.DataFrame, min_count: int = 3) -> pd.DataFrame:
    """Top directors/writers/cast by Bayesian-shrunk rating affinity, one row per (entity, role)."""
    affinity = compute_entity_affinity(ratings, metadata)
    if not affinity:
        return pd.DataFrame(columns=["entity", "role", "n_ratings", "avg_rating", "affinity"])
    rows = []
    for entity, info in affinity.items():
        if info["n_ratings"] < min_count:
            continue
        for role in info["columns"]:
            rows.append({
                "entity": entity,
                "role": role,
                "n_ratings": info["n_ratings"],
                "avg_rating": round(info["avg_rating"], 2),
                "affinity": round(info["affinity"], 3),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("affinity", ascending=False).reset_index(drop=True)


def feedback_summary(feedback: pd.DataFrame) -> pd.DataFrame:
    """Counts per taste-feedback label, with description and positive/neutral/negative polarity."""
    if feedback.empty:
        return pd.DataFrame(columns=["label", "description", "polarity", "count"])
    counts = feedback["feedback"].value_counts()
    rows = []
    for label, count in counts.items():
        info = FEEDBACK_LABELS.get(label)
        if info is None:
            continue
        weight = info["weight"]
        polarity = "positive" if weight > 0 else ("negative" if weight < 0 else "neutral")
        rows.append({"label": label, "description": info["description"], "polarity": polarity, "count": int(count)})
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def reflection_category_correlation(reflections: pd.DataFrame, min_films: int = 5) -> pd.DataFrame:
    """Per-category avg rating and Pearson correlation with your Overall rating."""
    categories = REFLECTION_CATEGORIES
    if reflections.empty:
        return pd.DataFrame(columns=["category", "avg_rating", "n_films", "corr_with_overall", "has_enough_data"])

    latest = (
        reflections.sort_values("created_at")
        .drop_duplicates(subset=["movie_id", "category"], keep="last")
    )
    pivot = latest.pivot_table(index="movie_id", columns="category", values="rating", aggfunc="last")

    rows = []
    if "Overall" not in pivot.columns:
        return pd.DataFrame(columns=["category", "avg_rating", "n_films", "corr_with_overall", "has_enough_data"])
    overall = pivot["Overall"]
    for category in categories:
        if category not in pivot.columns:
            continue
        paired = pd.concat([pivot[category], overall], axis=1, keys=["cat", "overall"]).dropna()
        n_films = len(paired)
        corr = paired["cat"].corr(paired["overall"]) if n_films >= 2 else float("nan")
        rows.append({
            "category": category,
            "avg_rating": round(pivot[category].dropna().mean(), 2) if pivot[category].notna().any() else float("nan"),
            "n_films": n_films,
            "corr_with_overall": round(corr, 3) if pd.notna(corr) else float("nan"),
            "has_enough_data": n_films >= min_films,
        })
    return pd.DataFrame(rows)


# Each external source's native scale, so everything can be rescaled to a common 0-10 axis
# for direct comparison against your own 0-5-star ratings (also rescaled to 0-10).
RATING_SOURCES = {
    "TMDb": {"column": "tmdb_vote_average", "scale": 10},
    "IMDb": {"column": "imdb_rating", "scale": 10},
    "Letterboxd": {"column": "letterboxd_rating", "scale": 5},
}


def mainstream_lean(ratings: pd.DataFrame, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Your rating vs. TMDb/IMDb/Letterboxd averages, all rescaled to the same 0-10 axis.

    Returns (per-movie comparison frame with a `your_rating_10` column plus
    `{source}_rating_10` / `{source}_delta` columns for whichever sources have
    data in `metadata`, excluded-count per source for rated films missing that source).
    """
    base_cols = ["Name", "Year", "your_rating_10"]
    if ratings.empty or metadata.empty:
        return pd.DataFrame(columns=base_cols), {s: 0 for s in RATING_SOURCES}

    r = ratings.copy()
    r["Rating"] = pd.to_numeric(r.get("Rating"), errors="coerce")
    r = r.dropna(subset=["Rating"])
    r["your_rating_10"] = r["Rating"] * 2

    available_sources = {s: cfg for s, cfg in RATING_SOURCES.items() if cfg["column"] in metadata.columns}
    meta_cols = ["movie_id"] + [cfg["column"] for cfg in available_sources.values()]
    merged = r.merge(metadata[meta_cols], on="movie_id", how="left") if available_sources else r.copy()

    excluded: Dict[str, int] = {}
    for source, cfg in available_sources.items():
        col = cfg["column"]
        excluded[source] = int(merged[col].isna().sum())
        rating_10_col = f"{source}_rating_10"
        merged[rating_10_col] = merged[col] * (10 / cfg["scale"])
        merged[f"{source}_delta"] = merged["your_rating_10"] - merged[rating_10_col]
    for source in RATING_SOURCES:
        if source not in available_sources:
            excluded[source] = len(merged)

    cols = base_cols + [f"{s}_rating_10" for s in available_sources] + [f"{s}_delta" for s in available_sources]
    return merged[cols].reset_index(drop=True), excluded


def list_affinity(lists: pd.DataFrame, ratings: pd.DataFrame, min_rated: int = 3) -> pd.DataFrame:
    """Which curated lists (source_list) you draw from most / rate highest."""
    if lists.empty:
        return pd.DataFrame(columns=["source_list", "n_entries", "n_rated", "avg_rating"])
    r = ratings.copy()
    r["Rating"] = pd.to_numeric(r.get("Rating"), errors="coerce")
    merged = lists.merge(r[["movie_id", "Rating"]], on="movie_id", how="left")
    grouped = merged.groupby("source_list").agg(
        n_entries=("movie_id", "count"),
        n_rated=("Rating", "count"),
        avg_rating=("Rating", "mean"),
    ).reset_index()
    grouped["avg_rating"] = grouped["avg_rating"].round(2)
    ranked = grouped[grouped["n_rated"] >= min_rated].sort_values("avg_rating", ascending=False)
    unranked = grouped[grouped["n_rated"] < min_rated].sort_values("n_entries", ascending=False)
    return pd.concat([ranked, unranked], ignore_index=True)
