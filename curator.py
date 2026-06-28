from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

logger = logging.getLogger(__name__)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recommender import prepare_metadata
import theme_similarity


DEFAULT_WEEK_SIZE = 7

CURATION_STYLES = [
    "Balanced",
    "Director-focused",
    "Theme-focused",
    "Vibe-focused",
    "Cinephile / historical context",
    "Gentler pacing",
]

BASE_ROLES = [
    "Context / influence",
    "Thematic setup",
    "Anchor movie",
    "Director / actor connection",
    "Intensifier",
    "Contrast / decompression",
    "Afterglow / reflection",
]

ROLE_DESCRIPTIONS = {
    "Context / influence": "Builds historical, genre, or cultural context before the anchor.",
    "Thematic setup": "Introduces similar themes, conflicts, or emotional territory.",
    "Anchor movie": "The centerpiece film the curation is built around.",
    "Director / actor connection": "Connects through director, writer, actor, or filmography.",
    "Intensifier": "Pushes the anchor's themes or mood further.",
    "Contrast / decompression": "Gives tonal contrast so the week does not become repetitive.",
    "Afterglow / reflection": "Ends the run with resonance, reflection, or emotional closure.",
    "Companion film": "A strong surrounding pick connected by theme, vibe, people, or context.",
}

STYLE_WEIGHTS = {
    "Balanced": {
        "similarity": 2.2,
        "director": 2.0,
        "cast": 0.8,
        "genres": 1.2,
        "keywords": 1.5,
        "theme_similarity": 1.5,
        "moods": 1.1,
        "decade": 0.5,
        "country": 0.4,
        "runtime_gentle": 0.0,
    },
    "Director-focused": {
        "similarity": 1.6,
        "director": 4.5,
        "cast": 1.0,
        "genres": 0.8,
        "keywords": 0.8,
        "theme_similarity": 0.8,
        "moods": 0.7,
        "decade": 0.4,
        "country": 0.3,
        "runtime_gentle": 0.0,
    },
    "Theme-focused": {
        "similarity": 2.4,
        "director": 1.0,
        "cast": 0.5,
        "genres": 1.0,
        "keywords": 3.0,
        "theme_similarity": 3.5,
        "moods": 1.2,
        "decade": 0.3,
        "country": 0.2,
        "runtime_gentle": 0.0,
    },
    "Vibe-focused": {
        "similarity": 2.0,
        "director": 0.8,
        "cast": 0.4,
        "genres": 1.2,
        "keywords": 1.4,
        "theme_similarity": 1.6,
        "moods": 3.0,
        "decade": 0.4,
        "country": 0.3,
        "runtime_gentle": 0.0,
    },
    "Cinephile / historical context": {
        "similarity": 1.8,
        "director": 1.7,
        "cast": 0.5,
        "genres": 1.0,
        "keywords": 1.4,
        "theme_similarity": 1.4,
        "moods": 0.8,
        "decade": 1.7,
        "country": 1.0,
        "runtime_gentle": 0.0,
    },
    "Gentler pacing": {
        "similarity": 1.8,
        "director": 1.2,
        "cast": 0.6,
        "genres": 1.0,
        "keywords": 1.1,
        "theme_similarity": 1.1,
        "moods": 1.1,
        "decade": 0.4,
        "country": 0.3,
        "runtime_gentle": 1.3,
    },
}


@dataclass(frozen=True)
class CuratorConfig:
    total_movies: int = DEFAULT_WEEK_SIZE
    style: str = "Balanced"
    allow_watched: bool = True
    allow_watchlisted: bool = True
    include_anchor: bool = True
    max_per_director: int = 2
    prefer_unwatched_bonus: float = 0.35


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if pd.notna(v)]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("[") and value.endswith("]"):
            import ast
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if pd.notna(v)]
            except Exception:
                logger.debug("ast.literal_eval failed on %r, falling back to comma split", value)
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _overlap(a: Any, b: Any) -> Tuple[int, List[str]]:
    left = set(_as_list(a))
    right = set(_as_list(b))
    matches = sorted(left & right)
    return len(matches), matches


def _decade(year: Any) -> str:
    try:
        if pd.isna(year):
            return "Unknown"
        y = int(year)
        return f"{y // 10 * 10}s"
    except Exception:
        return "Unknown"


def _safe_runtime(value: Any) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _role_sequence(total_movies: int, include_anchor: bool = True) -> List[str]:
    total_movies = max(1, int(total_movies or DEFAULT_WEEK_SIZE))
    if total_movies == 1:
        return ["Anchor movie"] if include_anchor else ["Companion film"]

    if total_movies <= len(BASE_ROLES):
        if include_anchor:
            # Keep the anchor near the middle for shorter runs.
            roles = [r for r in BASE_ROLES if r != "Anchor movie"]
            anchor_pos = min(total_movies - 1, max(1, total_movies // 2))
            selected = roles[:total_movies - 1]
            selected.insert(anchor_pos, "Anchor movie")
            return selected[:total_movies]
        return [r for r in BASE_ROLES if r != "Anchor movie"][:total_movies]

    roles = BASE_ROLES.copy() if include_anchor else [r for r in BASE_ROLES if r != "Anchor movie"]
    while len(roles) < total_movies:
        roles.insert(max(1, len(roles) - 1), "Companion film")
    return roles[:total_movies]


def _candidate_base(data: Dict[str, pd.DataFrame], metadata: pd.DataFrame, allow_watched: bool, allow_watchlisted: bool) -> pd.DataFrame:
    meta = prepare_metadata(metadata)
    if meta.empty:
        return pd.DataFrame()

    watched_ids = set(data.get("watched", pd.DataFrame()).get("movie_id", pd.Series(dtype=str)).dropna())
    rated_ids = set(data.get("ratings", pd.DataFrame()).get("movie_id", pd.Series(dtype=str)).dropna())
    watchlist_ids = set(data.get("watchlist", pd.DataFrame()).get("movie_id", pd.Series(dtype=str)).dropna())

    out = meta.copy()
    out["is_watched"] = out["movie_id"].isin(watched_ids | rated_ids)
    out["is_watchlisted"] = out["movie_id"].isin(watchlist_ids)
    if not allow_watched:
        out = out[~out["is_watched"]]
    if not allow_watchlisted:
        out = out[~out["is_watchlisted"]]
    return out.reset_index(drop=True)


def _movie_id_set(frame: pd.DataFrame) -> set[str]:
    if frame is None or frame.empty or "movie_id" not in frame.columns:
        return set()
    return {str(value) for value in frame["movie_id"].dropna().tolist()}


def _similarity_to_anchor(anchor: pd.Series, candidates: pd.DataFrame) -> pd.Series:
    if candidates.empty:
        return pd.Series(dtype=float)
    anchor_feature = anchor.get("feature_text", "")
    anchor_text = str(anchor_feature) if pd.notna(anchor_feature) else ""
    if "feature_text" in candidates.columns:
        candidate_text = candidates["feature_text"].fillna("").astype(str)
    else:
        candidate_text = pd.Series([""] * len(candidates), index=candidates.index, dtype="string")
    corpus = [anchor_text] + candidate_text.tolist()
    if not any(str(text).strip() for text in corpus):
        return pd.Series([0.0] * len(candidates), index=candidates.index)
    matrix = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=12000).fit_transform(corpus).toarray()
    sims = cosine_similarity(matrix[1:], matrix[:1]).ravel()
    return pd.Series(sims, index=candidates.index)


def _score_candidates(anchor: pd.Series, candidates: pd.DataFrame, config: CuratorConfig) -> pd.DataFrame:
    weights = STYLE_WEIGHTS.get(config.style, STYLE_WEIGHTS["Balanced"])
    out = candidates.copy()
    out["anchor_similarity"] = _similarity_to_anchor(anchor, out)
    # Conceptual/thematic similarity (keywords + overview, semantic when available),
    # complementing the literal keyword overlap below.
    anchor_id = str(anchor.get("movie_id"))
    theme_meta = pd.concat([out, anchor.to_frame().T], ignore_index=True)
    out["theme_sim"] = theme_similarity.theme_anchor_scores(out, anchor_id, theme_meta).to_numpy()

    anchor_decade = _decade(anchor.get("Year"))
    anchor_runtime = _safe_runtime(anchor.get("runtime"))

    rows = []
    for _, row in out.iterrows():
        director_n, director_matches = _overlap(row.get("directors", []), anchor.get("directors", []))
        cast_n, cast_matches = _overlap(row.get("cast", []), anchor.get("cast", []))
        genre_n, genre_matches = _overlap(row.get("genres", []), anchor.get("genres", []))
        keyword_n, keyword_matches = _overlap(row.get("keywords", []), anchor.get("keywords", []))
        mood_n, mood_matches = _overlap(row.get("moods", []), anchor.get("moods", []))
        country_n, country_matches = _overlap(row.get("countries", []), anchor.get("countries", []))

        decade_bonus = 1.0 if _decade(row.get("Year")) == anchor_decade and anchor_decade != "Unknown" else 0.0
        country_bonus = min(country_n, 2) / 2
        runtime = _safe_runtime(row.get("runtime"))
        gentle_bonus = 0.0
        if runtime is not None and runtime <= 130:
            gentle_bonus += 0.5
        if anchor_runtime is not None and runtime is not None and runtime < anchor_runtime:
            gentle_bonus += 0.4

        score = (
            row.get("anchor_similarity", 0) * weights["similarity"] * 5
            + row.get("theme_sim", 0) * weights.get("theme_similarity", 0)
            + min(director_n, 2) * weights["director"]
            + min(cast_n, 3) * weights["cast"]
            + min(genre_n, 3) * weights["genres"]
            + min(keyword_n, 5) * weights["keywords"]
            + min(mood_n, 3) * weights["moods"]
            + decade_bonus * weights["decade"]
            + country_bonus * weights["country"]
            + gentle_bonus * weights["runtime_gentle"]
        )
        if not bool(row.get("is_watched", False)):
            score += config.prefer_unwatched_bonus

        rows.append({
            "curation_score": float(score),
            "director_matches": director_matches,
            "cast_matches": cast_matches,
            "genre_matches": genre_matches,
            "keyword_matches": keyword_matches,
            "mood_matches": mood_matches,
            "country_matches": country_matches,
        })
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _reason_for(row: pd.Series, role: str, anchor_name: str) -> str:
    parts = []
    if row.get("director_matches"):
        parts.append("same director: " + ", ".join(row["director_matches"][:2]))
    if row.get("cast_matches"):
        parts.append("shared cast: " + ", ".join(row["cast_matches"][:2]))
    if row.get("keyword_matches"):
        parts.append("shared themes: " + ", ".join(row["keyword_matches"][:4]))
    if row.get("mood_matches"):
        parts.append("similar mood: " + ", ".join(row["mood_matches"][:3]))
    if row.get("genre_matches"):
        parts.append("genre overlap: " + ", ".join(row["genre_matches"][:3]))
    if not parts and float(row.get("anchor_similarity", 0) or 0) > 0:
        parts.append("metadata similarity to " + anchor_name)
    return "; ".join(parts) or ROLE_DESCRIPTIONS.get(role, "Companion pick for the anchor.")


def _pick_for_role(pool: pd.DataFrame, role: str, used_ids: set[str], director_counts: Dict[str, int], config: CuratorConfig) -> Optional[pd.Series]:
    if pool.empty:
        return None
    candidates = pool[~pool["movie_id"].isin(used_ids)].copy()
    if candidates.empty:
        return None

    role_bonus = pd.Series(0.0, index=candidates.index)
    if role == "Director / actor connection":
        role_bonus += candidates["director_matches"].apply(len) * 3.0 + candidates["cast_matches"].apply(len) * 0.8
    elif role == "Thematic setup":
        role_bonus += candidates["keyword_matches"].apply(len) * 1.3 + candidates["genre_matches"].apply(len) * 0.7
    elif role == "Context / influence":
        year_series = cast(pd.Series, pd.to_numeric(
            candidates["Year"] if "Year" in candidates.columns else pd.Series([pd.NA] * len(candidates), index=candidates.index),
            errors="coerce",
        ))
        role_bonus += candidates["country_matches"].apply(len) * 0.8
        role_bonus += year_series.rank(ascending=True, pct=True).fillna(0) * 0.4
    elif role == "Intensifier":
        role_bonus += candidates["mood_matches"].apply(len) * 1.1 + candidates["keyword_matches"].apply(len) * 0.8
    elif role == "Contrast / decompression":
        runtime = cast(pd.Series, pd.to_numeric(
            candidates["runtime"] if "runtime" in candidates.columns else pd.Series([pd.NA] * len(candidates), index=candidates.index),
            errors="coerce",
        ))
        role_bonus += runtime.apply(lambda x: 0.7 if pd.notna(x) and x <= 130 else 0.0)
    elif role == "Afterglow / reflection":
        moods = candidates.get("moods", pd.Series([[]] * len(candidates))).apply(_as_list)
        role_bonus += moods.apply(lambda vals: 0.8 if {"Reflective", "Emotional", "Light"} & set(vals) else 0.0)

    candidates["role_score"] = candidates["curation_score"] + role_bonus

    # Soft diversity: avoid too many films by the same director unless director-focused.
    for idx, row in candidates.iterrows():
        directors = _as_list(row.get("directors", []))
        if config.style != "Director-focused" and any(director_counts.get(d, 0) >= config.max_per_director for d in directors):
            candidates.loc[idx, "role_score"] -= 3.0

    sorted_candidates = cast(pd.DataFrame, candidates).sort_values(by="role_score", ascending=False)
    return cast(pd.Series, sorted_candidates.iloc[0])


def build_curated_list(
    anchor_movie_id: str,
    data: Dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    total_movies: int = DEFAULT_WEEK_SIZE,
    style: str = "Balanced",
    allow_watched: bool = True,
    allow_watchlisted: bool = True,
    include_anchor: bool = True,
) -> pd.DataFrame:
    """Build an ordered curated watchlist around one anchor movie.

    total_movies is the total number of rows in the final list. With include_anchor=True,
    the anchor counts as one of those movies. Default is 7.
    """
    config = CuratorConfig(
        total_movies=max(1, int(total_movies or DEFAULT_WEEK_SIZE)),
        style=style if style in STYLE_WEIGHTS else "Balanced",
        allow_watched=allow_watched,
        allow_watchlisted=allow_watchlisted,
        include_anchor=include_anchor,
    )
    meta = prepare_metadata(metadata)
    if meta.empty:
        return pd.DataFrame()

    anchor_rows = meta[meta["movie_id"] == anchor_movie_id]
    if anchor_rows.empty:
        # Also allow title labels such as "oldboy (2003)" if passed in.
        anchor_rows = meta[meta["movie_id"].astype(str).str.lower() == str(anchor_movie_id).lower()]
    if anchor_rows.empty:
        raise ValueError(f"Anchor movie not found in metadata: {anchor_movie_id}")

    anchor: pd.Series = anchor_rows.iloc[0]
    pool = cast(pd.DataFrame, _candidate_base(data, metadata, allow_watched=config.allow_watched, allow_watchlisted=config.allow_watchlisted))
    pool = cast(pd.DataFrame, pool[pool["movie_id"] != anchor["movie_id"]].copy())
    scored = _score_candidates(anchor, pool, config)

    roles = _role_sequence(config.total_movies, include_anchor=config.include_anchor)
    used_ids: set[str] = set()
    director_counts: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []

    def add_row(day: int, role: str, row: pd.Series, is_anchor: bool = False) -> None:
        mid = str(row.get("movie_id"))
        used_ids.add(mid)
        for director in _as_list(row.get("directors", [])):
            director_counts[director] = director_counts.get(director, 0) + 1
        rows.append({
            "day": day,
            "role": role,
            "Name": row.get("Name", row.get("name", "")),
            "Year": row.get("Year", row.get("year", "")),
            "movie_id": mid,
            "score": 999.0 if is_anchor else round(float(row.get("role_score", row.get("curation_score", 0)) or 0), 3),
            "why": "Anchor movie" if is_anchor else _reason_for(row, role, str(anchor.get("Name", "the anchor"))),
            "role_description": ROLE_DESCRIPTIONS.get(role, ROLE_DESCRIPTIONS["Companion film"]),
            "genres": row.get("genres", []),
            "moods": row.get("moods", []),
            "directors": row.get("directors", []),
            "runtime": row.get("runtime", ""),
            "poster_url": row.get("poster_url", ""),
            "overview": row.get("overview", ""),
            "tmdb_url": row.get("tmdb_url", ""),
        })

    for i, role in enumerate(roles, start=1):
        if role == "Anchor movie":
            add_row(i, role, anchor, is_anchor=True)
            continue
        pick = _pick_for_role(scored, role, used_ids, director_counts, config)
        if pick is not None:
            add_row(i, role, pick)

    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)


def anchor_options(metadata: pd.DataFrame, data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """Return movies that can be used as anchors, sorted for a Streamlit selectbox."""
    meta = prepare_metadata(metadata)
    if meta.empty:
        return pd.DataFrame(columns=["label", "movie_id", "Name", "Year", "source_labels"])

    base_cols = ["Name", "Year", "movie_id"]
    optional_cols = [
        column for column in ["poster_url", "genres", "moods", "directors", "overview", "tmdb_url"]
        if column in meta.columns
    ]
    opts = meta[base_cols + optional_cols].copy()

    if data is not None:
        watched_ids = _movie_id_set(data.get("watched", pd.DataFrame()))
        rated_ids = _movie_id_set(data.get("ratings", pd.DataFrame()))
        watchlist_ids = _movie_id_set(data.get("watchlist", pd.DataFrame()))

        def sources_for(movie_id: Any) -> List[str]:
            movie_id = str(movie_id)
            sources: List[str] = []
            if movie_id in watched_ids:
                sources.append("Watched")
            if movie_id in rated_ids:
                sources.append("Rated")
            if movie_id in watchlist_ids:
                sources.append("Watchlist")
            return sources

        opts["anchor_sources"] = opts["movie_id"].apply(sources_for)
        opts = opts[opts["anchor_sources"].apply(bool)].copy()
        opts["source_labels"] = opts["anchor_sources"].apply(lambda values: ", ".join(values))
    else:
        opts["anchor_sources"] = [[] for _ in range(len(opts))]
        opts["source_labels"] = ""

    opts["label"] = opts.apply(
        lambda row: f"{row['Name']} ({row['Year']})" + (f" [{row['source_labels']}]" if row.get("source_labels") else ""),
        axis=1,
    )
    sorted_opts = cast(pd.DataFrame, opts).sort_values(by=["Name", "Year"])
    return sorted_opts.drop_duplicates("movie_id").reset_index(drop=True)
