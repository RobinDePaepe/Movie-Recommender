from __future__ import annotations

import csv
import datetime
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

import theme_similarity

FEEDBACK_PATH = Path("data/feedback.csv")
LIST_SCORE_SCALE = 0.4
LIST_COUNT_WEIGHT = 0.15
# The heuristic (decade/recency/liked) is a tie-breaker, not the dominant signal:
# only its deviation from the 3.0 floor feeds the final score, damped by this weight.
HEURISTIC_WEIGHT = 0.5
# When a film is anchored, attenuate the personal-taste signals so the anchor leads
# (otherwise global taste cancels a concept-cousin anchor — the Moon/Primer problem).
ANCHOR_FOCUS_SCALE = 0.4

TASTE_MODES: Dict[str, Dict[str, Any]] = {
    "Balanced": {"terms": [], "runtime": None},
    "Comfort movie": {"terms": ["Light", "Comedy", "Family", "Animation", "Romance", "feel-good"], "runtime": (0, 130)},
    "Prestige drama": {"terms": ["Drama", "History", "Biography", "Reflective", "director", "novel"], "runtime": None},
    "Weird / arthouse": {"terms": ["Imaginative", "Fantasy", "Science Fiction", "surreal", "dream", "experimental"], "runtime": None},
    "Date night": {"terms": ["Romance", "Comedy", "Emotional", "Music", "Light"], "runtime": (0, 140)},
    "Short runtime": {"terms": [], "runtime": (0, 100)},
    "High confidence": {"terms": [], "runtime": None, "min_metadata": True},
}

# Feedback labels for taste tuning. Weights scale the content-similarity channel, so the
# sign steers recommendations toward (+) or away from (-) films with similar feature_text.
# Keys "more_like_this"/"less_like_this" are preserved for back-compat with saved feedback.
FEEDBACK_LABELS = {
    # Positive — pull toward similar films
    "all_time_favorite": {"weight": 2.5, "description": "All-time favorite"},
    "masterpiece": {"weight": 2.2, "description": "Masterpiece (rarely rewatched)"},
    "great_film": {"weight": 1.8, "description": "Great (rarely rewatched)"},
    "rewatchable": {"weight": 2.0, "description": "Rewatchable"},
    "more_like_this": {"weight": 1.5, "description": "More like this"},
    "pleasant_surprise": {"weight": 1.3, "description": "Pleasant surprise"},
    "comfort_watch": {"weight": 1.2, "description": "Comfort watch"},
    "guilty_pleasure": {"weight": 0.8, "description": "Guilty pleasure"},
    # Neutral — respect, but little or no pull
    "interesting_but_not_more": {"weight": 0.0, "description": "Interesting, but not more"},
    "mood_dependent": {"weight": 0.0, "description": "Mood-dependent"},
    "admire_not_love": {"weight": -0.3, "description": "Admire, don't love"},
    # Negative — push away from similar films
    "one_and_done": {"weight": -0.8, "description": "One and done"},
    "overrated_for_me": {"weight": -1.2, "description": "Overrated for me"},
    "high_quality_not_my_taste": {"weight": -1.5, "description": "High quality, not my taste"},
    "less_like_this": {"weight": -2.0, "description": "Less like this"},
    "actively_disliked": {"weight": -2.5, "description": "Actively disliked"},
}

# Deliberate watched-movie tuning is a stronger signal than passive feedback clicked from a
# recommendation. The multiplier scales each label's weight by where the feedback came from.
SCOPE_WEIGHTS = {
    "watched_tuning": 1.5,
    "recommendation": 1.0,
    "evaluation": 1.0,
}


def _scope_multiplier(feedback: pd.DataFrame) -> pd.Series:
    """Per-row scope multiplier; missing/unknown scopes default to 'recommendation' (1.0)."""
    if "scope" not in feedback.columns:
        return pd.Series(1.0, index=feedback.index)
    return feedback["scope"].map(SCOPE_WEIGHTS).fillna(SCOPE_WEIGHTS["recommendation"])


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def ensure_export_dir(export_zip: str | Path = "data/letterboxd_export.zip", out_dir: str | Path = "data/letterboxd") -> Path:
    export_zip = Path(export_zip)
    out_dir = Path(out_dir)
    if out_dir.exists() and (out_dir / "ratings.csv").exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(export_zip, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def normalize_movie_key(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["Name"] = out["Name"].astype(str).str.strip()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out["movie_id"] = out["Name"].str.lower() + " (" + out["Year"].astype(str) + ")"
    return out


def load_list_csv(path: Path) -> pd.DataFrame:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="latin-1").splitlines()
    header_index = None
    for i, line in enumerate(lines):
        if line.startswith("Position,Name,Year,URL"):
            header_index = i
            break
    if header_index is None:
        return pd.DataFrame()
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines[header_index:])))
    df["source_list"] = path.stem
    return normalize_movie_key(df)


def load_letterboxd(export_dir: str | Path) -> Dict[str, pd.DataFrame]:
    base = Path(export_dir)
    data = {
        "ratings": normalize_movie_key(_read_csv(base / "ratings.csv")),
        "watched": normalize_movie_key(_read_csv(base / "watched.csv")),
        "diary": normalize_movie_key(_read_csv(base / "diary.csv")),
        "watchlist": normalize_movie_key(_read_csv(base / "watchlist.csv")),
        "likes": normalize_movie_key(_read_csv(base / "likes" / "films.csv")),
    }
    list_frames = [load_list_csv(p) for p in (base / "lists").glob("*.csv")]
    data["lists"] = pd.concat([f for f in list_frames if not f.empty], ignore_index=True) if list_frames else pd.DataFrame()
    return data


def decade(year: float | int) -> str:
    if pd.isna(year):
        return "Unknown"
    y = int(year)
    return f"{y // 10 * 10}s"


def list_weight(list_name: str) -> float:
    name = str(list_name).lower()
    weights = 0.0
    rules = [
        (r"to-watch-this-week", 4.0),
        (r"watchlist|must-watch|must-watches", 2.3),
        (r"anticipated|critics", 1.8),
        (r"top|best|canon|essentials|masterworks|classics", 1.5),
        (r"rewatchables|ringer", 1.0),
        (r"owned", 0.6),
        (r"uhh", -1.0),
    ]
    for pattern, weight in rules:
        if re.search(pattern, name):
            weights += weight
    return weights


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
                pass
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def infer_moods(row: pd.Series) -> List[str]:
    genres = {g.lower() for g in _as_list(row.get("genres", []))}
    keywords = " ".join(_as_list(row.get("keywords", []))).lower()
    overview = str(row.get("overview", "") or "").lower()
    text = f"{keywords} {overview}"
    moods = set()
    if {"comedy", "family", "animation"} & genres:
        moods.add("Light")
    if {"horror", "thriller", "mystery"} & genres or any(w in text for w in ["serial killer", "haunting", "murder"]):
        moods.add("Tense")
    if {"drama", "romance"} & genres or any(w in text for w in ["grief", "heartbreak", "relationship"]):
        moods.add("Emotional")
    if {"action", "adventure", "war"} & genres:
        moods.add("Exciting")
    if {"science fiction", "fantasy"} & genres or any(w in text for w in ["surreal", "dream", "future"]):
        moods.add("Imaginative")
    if {"crime", "western"} & genres or any(w in text for w in ["revenge", "noir", "gangster"]):
        moods.add("Gritty")
    if {"documentary", "history"} & genres:
        moods.add("Reflective")
    return sorted(moods) or ["General"]


def _feature_text(row: pd.Series) -> str:
    fields: List[str] = []
    weighted_fields = {"genres": 4, "directors": 5, "writers": 2, "cast": 2, "keywords": 3, "countries": 1, "languages": 1, "moods": 2}
    for col, weight in weighted_fields.items():
        fields.extend(_as_list(row.get(col, [])) * weight)
    fields.append(str(row.get("overview", "") or ""))
    return " ".join(fields).lower()


def prepare_metadata(metadata: pd.DataFrame | None) -> pd.DataFrame:
    if metadata is None or metadata.empty:
        return pd.DataFrame()
    meta = metadata.copy()
    name_col = "name" if "name" in meta.columns else "Name"
    year_col = "year" if "year" in meta.columns else "Year"
    meta["Name"] = meta[name_col].astype(str)
    meta["Year"] = pd.to_numeric(meta.get(year_col), errors="coerce").astype("Int64")
    meta = normalize_movie_key(meta)
    if "moods" not in meta.columns:
        meta["moods"] = meta.apply(infer_moods, axis=1)
    if "poster_url" not in meta.columns and "poster_path" in meta.columns:
        meta["poster_url"] = meta["poster_path"].fillna("").apply(lambda p: f"https://image.tmdb.org/t/p/w342{p}" if p else "")
    meta["feature_text"] = meta.apply(_feature_text, axis=1)
    return meta.drop_duplicates("movie_id")


def load_feedback(path: str | Path = FEEDBACK_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["movie_id", "feedback", "scope"])
    df = pd.read_csv(path)
    if "scope" not in df.columns:
        df["scope"] = "recommendation"
    else:
        df["scope"] = df["scope"].fillna("recommendation")
    return df


def save_feedback(movie_id: str, feedback: str, scope: str = "recommendation", path: str | Path = FEEDBACK_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_feedback(path)
    if not existing.empty and ((existing["movie_id"] == movie_id) & (existing["feedback"] == feedback)).any():
        return
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["movie_id", "feedback", "scope"])
        if not exists:
            writer.writeheader()
        writer.writerow({"movie_id": movie_id, "feedback": feedback, "scope": scope})


def remove_feedback_from_csv(movie_id: str, labels: list, path: str | Path = FEEDBACK_PATH) -> None:
    path = Path(path)
    df = load_feedback(path)
    if df.empty:
        return
    df = df[~((df["movie_id"] == movie_id) & (df["feedback"].isin(labels)))]
    df.to_csv(path, index=False)


def candidate_pool(data: Dict[str, pd.DataFrame], mode: str = "watchlist") -> pd.DataFrame:
    watched_ids = set(data["watched"].get("movie_id", pd.Series(dtype=str)).dropna())
    watchlist = data["watchlist"].copy()
    watchlist_ids = set(watchlist.get("movie_id", pd.Series(dtype=str)).dropna())
    if mode == "outside_watchlist":
        frames = []
        if not data["lists"].empty:
            frames.append(data["lists"][["Name", "Year", "URL", "movie_id"]].rename(columns={"URL": "Letterboxd URI"}))
        pool = pd.concat(frames, ignore_index=True).drop_duplicates("movie_id") if frames else pd.DataFrame(columns=["Name", "Year", "movie_id", "Letterboxd URI"])
        return pool[~pool["movie_id"].isin(watched_ids | watchlist_ids)].copy()
    return watchlist[~watchlist["movie_id"].isin(watched_ids)].copy()


def add_heuristic_scores(candidates: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = data["ratings"].copy()
    likes = data["likes"].copy()
    lists = data["lists"].copy()
    ratings["Rating"] = pd.to_numeric(ratings.get("Rating"), errors="coerce")
    ratings["decade"] = ratings["Year"].apply(decade)
    global_mean = ratings["Rating"].mean() if not ratings.empty else 3.0
    # Bayesian-shrink per-decade means toward the global mean so one lucky 5★ film
    # in a sparse decade doesn't inflate that decade's score (mirrors entity affinity).
    DECADE_PRIOR_COUNT = 3
    decade_pref = ratings.groupby("decade")["Rating"].agg(["mean", "count"]).reset_index()
    decade_pref["avg_user_rating"] = decade_pref["mean"]
    decade_pref["bayes"] = (
        decade_pref["mean"] * decade_pref["count"] + global_mean * DECADE_PRIOR_COUNT
    ) / (decade_pref["count"] + DECADE_PRIOR_COUNT)
    decade_pref["decade_score"] = (decade_pref["bayes"] - global_mean).fillna(0) * 1.2

    out = candidates.copy()
    out["decade"] = out["Year"].apply(decade)
    out = out.merge(decade_pref[["decade", "decade_score", "avg_user_rating"]], on="decade", how="left")
    out["decade_score"] = out["decade_score"].fillna(0)
    if not lists.empty:
        lists["list_signal"] = lists["source_list"].apply(list_weight)
        list_features = lists.groupby("movie_id").agg(
            list_score=("list_signal", "sum"),
            list_count=("source_list", "nunique"),
            list_names=("source_list", lambda s: ", ".join(sorted(set(s))[:6])),
        ).reset_index()
        out = out.merge(list_features, on="movie_id", how="left")
    else:
        out["list_score"] = 0
        out["list_count"] = 0
        out["list_names"] = ""
    liked_decades = set(likes["Year"].apply(decade)) if not likes.empty else set()
    out["liked_decade_bonus"] = out["decade"].apply(lambda d: 0.7 if d in liked_decades else 0.0)
    current_year = datetime.date.today().year
    # Recency bonus fades linearly from 0.8 (current year) to 0 over a 15-year window,
    # instead of a hardcoded cliff that goes stale as years pass.
    out["recency_bonus"] = pd.to_numeric(out["Year"], errors="coerce").apply(
        lambda y: max(0.0, 0.8 * (1 - (current_year - y) / 15)) if pd.notna(y) else 0.0
    )
    out[["list_score", "list_count"]] = out[["list_score", "list_count"]].fillna(0)
    out["list_names"] = out["list_names"].fillna("")
    out["list_contribution"] = (out["list_score"] * LIST_SCORE_SCALE) + (out["list_count"].clip(upper=5) * LIST_COUNT_WEIGHT)
    out["heuristic_score"] = 3.0 + out["decade_score"] + out["liked_decade_bonus"] + out["recency_bonus"] + out["list_contribution"]
    return out, decade_pref.sort_values("decade")


def build_taste_profile(positive_meta: pd.DataFrame, top_n: int = 12) -> Dict[str, List[str]]:
    profile: Dict[str, List[str]] = {}
    for col in ["genres", "directors", "cast", "keywords", "countries", "languages", "moods"]:
        counts: Dict[str, int] = {}
        for values in positive_meta.get(col, pd.Series(dtype=object)):
            for value in _as_list(values):
                counts[value] = counts.get(value, 0) + 1
        profile[col] = [item for item, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]
    return profile


def taste_match_text(row: pd.Series, profile: Dict[str, List[str]]) -> str:
    labels = {"genres": "genres", "directors": "directors", "cast": "actors", "keywords": "themes", "countries": "countries", "languages": "languages", "moods": "moods"}
    parts = []
    for col, label in labels.items():
        row_values = set(_as_list(row.get(col, [])))
        matches = [v for v in profile.get(col, []) if v in row_values][:3]
        if matches:
            parts.append(f"{label}: {', '.join(matches)}")
    return "; ".join(parts)


def _rating_weight(rating: float) -> float:
    """Convert a star rating into a similarity weight. 5★ → 1.0, 4.5★ → 0.67, 4.0★ → 0.33."""
    return max(0.1, (float(rating) - 3.0) / 2.0)


def add_content_similarity(candidates: pd.DataFrame, ratings: pd.DataFrame, likes: pd.DataFrame, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    import numpy as np
    meta = prepare_metadata(metadata)
    if meta.empty:
        candidates["content_similarity"] = 0.0
        candidates["content_score"] = 0.0
        candidates["metadata_found"] = False
        candidates["taste_matches"] = ""
        return candidates, {}
    rated = ratings.copy()
    rated["Rating"] = pd.to_numeric(rated.get("Rating"), errors="coerce")
    liked_ids = set(likes.get("movie_id", pd.Series(dtype=str)).dropna())
    positive_ids = set(rated.loc[rated["Rating"] >= 4.0, "movie_id"].dropna()) | liked_ids
    negative_ids = set(rated.loc[rated["Rating"] <= 2.5, "movie_id"].dropna())
    positive_meta = meta[meta["movie_id"].isin(positive_ids) & meta["feature_text"].str.len().gt(0)].copy()
    negative_meta = meta[meta["movie_id"].isin(negative_ids) & meta["feature_text"].str.len().gt(0)].copy()
    cand = candidates.merge(meta.drop(columns=["Name", "Year"], errors="ignore"), on="movie_id", how="left", suffixes=("", "_tmdb"))
    cand["metadata_found"] = cand.get("tmdb_found", False).fillna(False).astype(bool) if "tmdb_found" in cand.columns else False
    cand["feature_text"] = cand.get("feature_text", "").fillna("")
    if positive_meta.empty or cand["feature_text"].str.len().sum() == 0:
        cand["content_similarity"] = 0.0
        cand["content_score"] = 0.0
        cand["taste_matches"] = ""
        return cand, {}

    # Build a single corpus so all vectors share the same TF-IDF space
    n_pos = len(positive_meta)
    n_neg = len(negative_meta)
    corpus = (
        positive_meta["feature_text"].tolist()
        + negative_meta["feature_text"].tolist()
        + cand["feature_text"].tolist()
    )
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=12000)
    matrix = vectorizer.fit_transform(corpus)
    pos_matrix = matrix[:n_pos]
    neg_matrix = matrix[n_pos:n_pos + n_neg]
    cand_matrix = matrix[n_pos + n_neg:]

    # Rating-weighted positive similarity: 5★ films pull harder than 4★ films
    rating_lookup = rated.dropna(subset=["Rating"]).set_index("movie_id")["Rating"]
    pos_weights = np.array([
        _rating_weight(rating_lookup[mid]) if mid in rating_lookup.index else _rating_weight(4.0)
        for mid in positive_meta["movie_id"]
    ])
    pos_sims = cosine_similarity(cand_matrix, pos_matrix)
    weight_sum = pos_weights.sum()
    weighted_pos_sim = (pos_sims * pos_weights).sum(axis=1) / weight_sum if weight_sum > 0 else pos_sims.mean(axis=1)

    # Negative penalty: penalise candidates similar to films you disliked
    if n_neg > 0:
        neg_sims = cosine_similarity(cand_matrix, neg_matrix)
        neg_penalty = neg_sims.mean(axis=1) * 1.5
    else:
        neg_penalty = np.zeros(len(cand))

    cand["content_similarity"] = weighted_pos_sim - neg_penalty
    max_sim = cand["content_similarity"].max()
    cand["content_score"] = (cand["content_similarity"] / max_sim * 4.0) if max_sim and max_sim > 0 else 0.0
    taste_profile = build_taste_profile(positive_meta)
    cand["taste_matches"] = cand.apply(lambda row: taste_match_text(row, taste_profile), axis=1)
    return cand, taste_profile


def add_entity_affinity(candidates: pd.DataFrame, ratings: pd.DataFrame, metadata: pd.DataFrame | None) -> pd.DataFrame:
    """Score candidates by your historical ratings for their directors, cast, and writers.

    Uses a Bayesian average (pulled toward the global mean) so one lucky 5★ film
    doesn't inflate an entity's score. Directors are weighted most heavily.
    """
    import numpy as np
    out = candidates.copy()
    out["entity_score"] = 0.0
    if ratings.empty or metadata is None:
        return out
    meta = prepare_metadata(metadata)
    if meta.empty:
        return out
    rated = ratings.copy()
    rated["Rating"] = pd.to_numeric(rated.get("Rating"), errors="coerce")
    rated = rated.dropna(subset=["Rating"])
    rated_meta = rated.merge(meta[["movie_id", "directors", "cast", "writers"]], on="movie_id", how="inner")
    if rated_meta.empty:
        return out

    global_mean = rated_meta["Rating"].mean()
    PRIOR_COUNT = 3  # shrink small samples toward the mean
    COL_WEIGHTS = {"directors": 1.5, "writers": 0.8, "cast": 0.4}

    # Collect ratings per entity across all tracked columns
    entity_ratings: Dict[str, List[float]] = {}
    for _, row in rated_meta.iterrows():
        for col in COL_WEIGHTS:
            for entity in _as_list(row.get(col, [])):
                entity_ratings.setdefault(entity, []).append(float(row["Rating"]))

    # Bayesian average → deviation from global mean → weighted affinity score
    entity_affinity: Dict[str, float] = {}
    for entity, rs in entity_ratings.items():
        n = len(rs)
        bayes_avg = (sum(rs) + PRIOR_COUNT * global_mean) / (n + PRIOR_COUNT)
        entity_affinity[entity] = (bayes_avg - global_mean) / 2.0  # normalised deviation

    def calc_entity_score(row: pd.Series) -> float:
        score = 0.0
        for col, col_w in COL_WEIGHTS.items():
            entities = _as_list(row.get(col, []))
            if not entities:
                continue
            # Best match per column — don't sum so a large cast doesn't game the score
            best = max((entity_affinity.get(e, 0.0) for e in entities), default=0.0)
            score += best * col_w
        return float(np.clip(score, -2.0, 2.0))

    out["entity_score"] = out.apply(calc_entity_score, axis=1)
    return out


def add_feedback_similarity(candidates: pd.DataFrame, feedback: pd.DataFrame, metadata: pd.DataFrame | None) -> pd.DataFrame:
    out = candidates.copy()
    out["feedback_score"] = 0.0
    if feedback is None or feedback.empty or metadata is None or metadata.empty:
        return out
    meta = prepare_metadata(metadata)
    if meta.empty or "feature_text" not in out.columns:
        return out
    fb = feedback.merge(meta[["movie_id", "feature_text"]], on="movie_id", how="inner")
    fb = fb[fb["feature_text"].str.len().gt(0)]
    cand_idx = out["feature_text"].fillna("").str.len().gt(0)
    if fb.empty or not cand_idx.any():
        return out
    corpus = fb["feature_text"].tolist() + out.loc[cand_idx, "feature_text"].fillna("").tolist()
    matrix = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=12000).fit_transform(corpus)
    sims = cosine_similarity(matrix[len(fb):], matrix[:len(fb)])
    label_w = {k: v["weight"] for k, v in FEEDBACK_LABELS.items()}
    weights = (fb["feedback"].map(label_w).fillna(0) * _scope_multiplier(fb)).to_numpy()
    out.loc[cand_idx, "feedback_score"] = (sims @ weights).clip(-3.0, 3.0)
    direct = feedback["feedback"].map(label_w).fillna(0) * _scope_multiplier(feedback)
    direct_adj = feedback.assign(direct_score=direct).groupby("movie_id", as_index=False)["direct_score"].sum()
    out = out.merge(direct_adj, on="movie_id", how="left")
    out["feedback_score"] = out["feedback_score"].fillna(0) + out["direct_score"].fillna(0)
    return out.drop(columns=["direct_score"], errors="ignore")


def taste_mode_score(row: pd.Series, taste_mode: str) -> float:
    config = TASTE_MODES.get(taste_mode, TASTE_MODES["Balanced"])
    score = 0.0
    text = " ".join(_as_list(row.get("genres", [])) + _as_list(row.get("moods", [])) + _as_list(row.get("keywords", []))).lower()
    for term in config.get("terms", []):
        if str(term).lower() in text:
            score += 0.35
    rt_range = config.get("runtime")
    runtime = pd.to_numeric(row.get("runtime"), errors="coerce")
    if rt_range and pd.notna(runtime) and rt_range[0] <= runtime <= rt_range[1]:
        score += 0.8
    if config.get("min_metadata") and row.get("metadata_found") and row.get("content_score", 0) >= 2.0:
        score += 1.0
    return min(score, 2.0)


def explain_short(row: pd.Series, taste_mode: str = "Balanced") -> str:
    parts: List[str] = []
    cs = float(row.get("content_score", 0) or 0)
    fb = float(row.get("feedback_score", 0) or 0)
    if float(row.get("anchor_score", 0) or 0) > 0.5:
        parts.append("Matches anchor film")
    if cs >= 3.0:
        parts.append("Strong taste match")
    elif cs > 0.75:
        parts.append("Matches taste profile")
    if float(row.get("theme_score", 0) or 0) >= 2.0:
        parts.append("Thematically similar")
    if abs(fb) > 0.2:
        parts.append("Taste feedback: positive" if fb > 0 else "Taste feedback: negative")
    if float(row.get("taste_mode_score", 0) or 0) > 0:
        parts.append(taste_mode)
    if float(row.get("entity_score", 0) or 0) > 0.3:
        parts.append("Trusted director/cast")
    if int(row.get("list_count", 0) or 0) > 0:
        parts.append(f"In {int(row.get('list_count'))} lists")
    if row.get("decade_score", 0) > 0.2:
        parts.append("Decade affinity")
    if row.get("recency_bonus", 0) > 0:
        parts.append("Recent")
    if row.get("metadata_found") is False and cs == 0:
        parts.append("No metadata")
    return "; ".join(parts) or "Solid candidate"


def explain_detailed(row: pd.Series, taste_mode: str = "Balanced") -> str:
    reasons = []
    if float(row.get("anchor_score", 0) or 0) > 0.5:
        reasons.append("thematically similar to your chosen anchor film")
    if row.get("content_score", 0) > 0.75:
        reasons.append("matches your high-rated taste profile: {}".format(row.get("taste_matches", "")))
    if float(row.get("theme_score", 0) or 0) >= 1.5:
        reasons.append("explores themes/concepts like the films you rate highly")
    if abs(float(row.get("feedback_score", 0) or 0)) > 0.2:
        reasons.append("adjusted by similarity to movies you tagged with taste feedback")
    entity = float(row.get("entity_score", 0) or 0)
    if entity > 0.3:
        reasons.append("directed/written/starring someone you've consistently rated highly")
    elif entity < -0.3:
        reasons.append("involves someone you've rated poorly in the past")
    if float(row.get("taste_mode_score", 0) or 0) > 0:
        reasons.append(f"fits the selected taste mode: {taste_mode}")
    if row.get("list_count", 0) > 0:
        reasons.append(f"appears in {int(row.get('list_count', 0))} lists: {row.get('list_names', '')}")
    if row.get("decade_score", 0) > 0.2:
        reasons.append("matches your decade preferences")
    if row.get("liked_decade_bonus", 0) > 0:
        reasons.append("same decade as films you liked")
    if row.get("recency_bonus", 0) > 0:
        reasons.append("recent release bonus")
    if row.get("metadata_found") is False and row.get("content_score", 0) == 0:
        reasons.append("no TMDb metadata available")
    return "; ".join([r for r in reasons if r]) or "solid candidate"


def add_anchor_similarity(candidates: pd.DataFrame, anchor_movie_id: str | None, metadata: pd.DataFrame | None) -> pd.DataFrame:
    """Boost candidates that are *thematically* similar to one anchor film.

    Now delegates to theme_similarity so "similar to Inception" means "explores the same
    ideas" (dreams, bent reality) rather than "same director / genre / cast". Scaled to 4.0
    (on par with content_score) so an explicit anchor is a primary signal; tunable via the
    anchor weight.
    """
    out = candidates.copy()
    out["anchor_score"] = 0.0
    if not anchor_movie_id or metadata is None:
        return out
    meta = prepare_metadata(metadata)
    if meta.empty or "movie_id" not in out.columns:
        return out
    out["anchor_score"] = theme_similarity.theme_anchor_scores(out, anchor_movie_id, meta)
    return out


def add_theme_similarity(candidates: pd.DataFrame, ratings: pd.DataFrame, likes: pd.DataFrame | None, metadata: pd.DataFrame | None) -> pd.DataFrame:
    """Standalone 'what a film is about' channel vs the themes of your high-rated films.

    Distinct from content_score (which is dominated by genre/director/cast); this measures
    conceptual/thematic closeness only (keywords + overview).
    """
    out = candidates.copy()
    out["theme_score"] = 0.0
    if metadata is None:
        return out
    meta = prepare_metadata(metadata)
    if meta.empty or "movie_id" not in out.columns:
        return out
    out["theme_score"] = theme_similarity.theme_taste_scores(
        out, ratings, likes if likes is not None else pd.DataFrame(), meta
    )
    return out


def apply_mood_avoidance(candidates: pd.DataFrame, avoid_moods: List[str], penalty: float = 1.5) -> pd.DataFrame:
    """Subtract a score penalty for candidates whose moods overlap with the avoid list."""
    if not avoid_moods:
        candidates = candidates.copy()
        candidates["mood_penalty"] = 0.0
        return candidates
    avoid_set = set(avoid_moods)
    out = candidates.copy()
    mask = out.apply(lambda row: bool(avoid_set & set(_as_list(row.get("moods", [])))), axis=1)
    out["mood_penalty"] = 0.0
    out.loc[mask, "mood_penalty"] = penalty
    out["score"] = out["score"] - out["mood_penalty"]
    return out


def build_recommendations(data: Dict[str, pd.DataFrame], metadata: pd.DataFrame | None = None, mode: str = "watchlist", feedback: pd.DataFrame | None = None, taste_mode: str = "Balanced", score_weights: Dict[str, float] | None = None, anchor_movie_id: str | None = None, avoid_moods: List[str] | None = None, anchor_focus: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidates = candidate_pool(data, mode=mode)
    meta = prepare_metadata(metadata)
    if mode == "outside_watchlist" and not meta.empty:
        watched_ids = set(data["watched"].get("movie_id", pd.Series(dtype=str)).dropna())
        rated_ids = set(data["ratings"].get("movie_id", pd.Series(dtype=str)).dropna())
        watchlist_ids = set(data["watchlist"].get("movie_id", pd.Series(dtype=str)).dropna())
        known_ids = set(candidates.get("movie_id", pd.Series(dtype=str)).dropna())
        outside = meta[~meta["movie_id"].isin(watched_ids | rated_ids | watchlist_ids | known_ids)]
        # Scope the metadata-sourced pool to genuinely *discovered* films (those carrying
        # `discovered_from` provenance), so cache entries fetched only to enrich the watchlist
        # don't leak into "Not on my watchlist". Fall back to all cached rows only when the
        # cache predates the column entirely, so older caches aren't left with an empty pool.
        if "discovered_from" in outside.columns:
            provenance = outside["discovered_from"].fillna("").astype(str).str.strip()
            outside = outside[provenance != ""]
        extra = outside[["Name", "Year", "movie_id", "tmdb_url"]].copy()
        extra["Letterboxd URI"] = ""
        candidates = pd.concat([candidates, extra], ignore_index=True, sort=False).drop_duplicates("movie_id")
    candidates, decade_prefs = add_heuristic_scores(candidates, data)
    candidates, taste_profile = add_content_similarity(candidates, data["ratings"], data["likes"], metadata if metadata is not None else pd.DataFrame())
    candidates = add_feedback_similarity(candidates, feedback if feedback is not None else pd.DataFrame(), metadata)
    candidates["taste_mode_score"] = candidates.apply(lambda row: taste_mode_score(row, taste_mode), axis=1)
    candidates = add_entity_affinity(candidates, data["ratings"], metadata)
    candidates = add_anchor_similarity(candidates, anchor_movie_id, metadata)
    candidates = add_theme_similarity(candidates, data["ratings"], data["likes"], metadata)

    weights = score_weights or {}
    content_w = float(weights.get("content", 1.0))
    entity_w = float(weights.get("entity", 1.0))
    list_w = float(weights.get("list", 1.0))
    anchor_w = float(weights.get("anchor", 1.0))
    theme_w = float(weights.get("theme", 1.0))
    feedback_w = float(weights.get("feedback", 1.0))
    # Anchor-focus: when a film is anchored, ease off the personal-taste signals so the
    # anchor leads ("show me films like THIS"), instead of global taste cancelling it.
    if anchor_movie_id and anchor_focus:
        content_w *= ANCHOR_FOCUS_SCALE
        theme_w *= ANCHOR_FOCUS_SCALE
        entity_w *= ANCHOR_FOCUS_SCALE
    # Strip the 3.0 floor and list_contribution (re-added below with its own weight) so
    # the heuristic contributes only its deviation, damped to a tie-breaker.
    base_heuristic = (candidates["heuristic_score"] - candidates["list_contribution"] - 3.0) * HEURISTIC_WEIGHT
    candidates["score"] = (
        base_heuristic
        + candidates["list_contribution"] * list_w
        + candidates["content_score"] * content_w
        + candidates["feedback_score"] * feedback_w
        + candidates["taste_mode_score"]
        + candidates["entity_score"] * entity_w
        + candidates["anchor_score"] * anchor_w
        + candidates["theme_score"] * theme_w
    )
    candidates = apply_mood_avoidance(candidates, avoid_moods or [])
    candidates["list_names_full"] = candidates.get("list_names", pd.Series(dtype=str)).fillna("").astype(str)
    candidates["taste_matches_full"] = candidates.get("taste_matches", pd.Series(dtype=str)).fillna("").astype(str)
    candidates["list_names"] = candidates["list_names_full"].apply(lambda s: s if len(s) <= 140 else s[:137] + "...")
    candidates["taste_matches"] = candidates["taste_matches_full"].apply(lambda s: s if len(s) <= 140 else s[:137] + "...")
    candidates["why_details"] = candidates.apply(lambda row: explain_detailed(row, taste_mode), axis=1)
    candidates["why"] = candidates.apply(lambda row: explain_short(row, taste_mode), axis=1)
    cols = ["Name", "Year", "score", "heuristic_score", "list_contribution", "content_similarity", "content_score", "feedback_score", "taste_mode_score", "entity_score", "anchor_score", "theme_score", "mood_penalty", "why", "why_details", "Letterboxd URI", "movie_id", "decade", "list_names", "taste_matches", "list_names_full", "taste_matches_full"]
    for optional_col in ["genres", "moods", "runtime", "languages", "directors", "cast", "keywords", "tmdb_url", "poster_url", "overview", "tmdb_vote_average", "tmdb_popularity", "discovered_from"]:
        if optional_col in candidates.columns:
            cols.append(optional_col)
    for c in cols:
        if c not in candidates.columns:
            candidates[c] = ""
    return candidates.sort_values("score", ascending=False)[cols].reset_index(drop=True), decade_prefs


def available_filter_values(recs: pd.DataFrame) -> Dict[str, List[str]]:
    values: Dict[str, List[str]] = {}
    for col in ["genres", "languages", "moods"]:
        found = set()
        if col in recs.columns:
            for item in recs[col]:
                found.update(_as_list(item))
        values[col] = sorted(found)
    values["decades"] = sorted([d for d in recs.get("decade", pd.Series(dtype=str)).dropna().unique().tolist() if d != "Unknown"])
    values["taste_modes"] = list(TASTE_MODES.keys())
    return values


def apply_filters(recs: pd.DataFrame, genres=None, languages=None, moods=None, decades=None, runtime_range=None, query: str = "") -> pd.DataFrame:
    filtered = recs.copy()
    for col, selected in [("genres", genres), ("languages", languages), ("moods", moods)]:
        selected = set(selected or [])
        if selected and col in filtered.columns:
            filtered = filtered[filtered[col].apply(lambda vals: bool(selected & set(_as_list(vals))))]
    if decades:
        filtered = filtered[filtered["decade"].isin(decades)]
    if runtime_range and "runtime" in filtered.columns:
        runtime = pd.to_numeric(filtered["runtime"], errors="coerce")
        filtered = filtered[(runtime.isna()) | ((runtime >= runtime_range[0]) & (runtime <= runtime_range[1]))]
    if query:
        q = query.lower()
        searchable_cols = ["Name", "list_names", "why", "taste_matches", "genres", "directors", "cast", "keywords", "moods", "languages"]
        existing_cols = [c for c in searchable_cols if c in filtered.columns]
        filtered = filtered[filtered[existing_cols].astype(str).apply(lambda row: q in " ".join(row).lower(), axis=1)]
    return filtered


def evaluate_historical_predictions(data: Dict[str, pd.DataFrame], metadata: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    meta = prepare_metadata(metadata)
    ratings = data["ratings"].copy()
    if meta.empty or ratings.empty:
        return pd.DataFrame(), {}
    ratings["Rating"] = pd.to_numeric(ratings.get("Rating"), errors="coerce")
    rated_meta = ratings.merge(meta.drop(columns=["Name", "Year"], errors="ignore"), on="movie_id", how="inner")
    rated_meta = rated_meta.dropna(subset=["Rating"])
    rated_meta = rated_meta[rated_meta["feature_text"].str.len().gt(0)].copy()
    if len(rated_meta) < 20:
        return pd.DataFrame(), {"error": "Need at least 20 rated movies with TMDb metadata for evaluation."}
    rated_meta = rated_meta.sort_values("movie_id").reset_index(drop=True)
    rated_meta["is_test"] = rated_meta.index % 5 == 0
    train = rated_meta[~rated_meta["is_test"]]
    test = rated_meta[rated_meta["is_test"]].copy()
    positive_train = train[train["Rating"] >= 4.0]
    if positive_train.empty or test.empty:
        return pd.DataFrame(), {}
    corpus = positive_train["feature_text"].tolist() + test["feature_text"].tolist()
    matrix = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=12000).fit_transform(corpus)
    sims = cosine_similarity(matrix[len(positive_train):], matrix[:len(positive_train)])
    test["predicted_similarity"] = sims.mean(axis=1) if sims.size else 0.0
    min_sim, max_sim = test["predicted_similarity"].min(), test["predicted_similarity"].max()
    test["predicted_rating"] = 1.0 + ((test["predicted_similarity"] - min_sim) / (max_sim - min_sim) * 4.0) if max_sim > min_sim else train["Rating"].mean()
    test = test.sort_values("predicted_similarity", ascending=False).reset_index(drop=True)
    test["relevant"] = test["Rating"] >= 4.0
    def precision_at(k: int) -> float:
        top = test.head(k)
        return float(top["relevant"].mean()) if len(top) else 0.0
    def recall_at(k: int) -> float:
        total_rel = int(test["relevant"].sum())
        return float(test.head(k)["relevant"].sum() / total_rel) if total_rel else 0.0
    def dcg(vals: pd.Series) -> float:
        import math
        return float(sum((2 ** v - 1) / math.log2(i + 2) for i, v in enumerate(vals)))
    gains = test["Rating"].fillna(0)
    ideal = gains.sort_values(ascending=False)
    ndcg10 = dcg(gains.head(10)) / dcg(ideal.head(10)) if dcg(ideal.head(10)) else 0.0
    metrics = {
        "rated_movies_with_metadata": float(len(rated_meta)),
        "test_movies": float(len(test)),
        "mae": float(mean_absolute_error(test["Rating"], test["predicted_rating"])),
        "correlation": float(test[["Rating", "predicted_similarity"]].corr().iloc[0, 1]) if len(test) > 2 else 0.0,
        "precision_at_10": precision_at(10),
        "recall_at_25": recall_at(25),
        "ndcg_at_10": ndcg10,
        "top20_4star_hits": float(test.head(20)["relevant"].sum()),
    }
    out_cols = ["Name", "Year", "Rating", "predicted_rating", "predicted_similarity", "genres", "directors", "keywords"]
    return test[out_cols], metrics


if __name__ == "__main__":
    export_dir = ensure_export_dir()
    data = load_letterboxd(export_dir)
    recs, prefs = build_recommendations(data)
    print(recs.head(25).to_string(index=False))
