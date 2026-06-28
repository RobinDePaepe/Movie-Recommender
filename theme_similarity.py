"""Thematic / conceptual similarity.

This module isolates *what a film is about* (its keywords + overview) from the
surface metadata (genre / director / cast) that dominates ``recommender._feature_text``.
That "aboutness" axis is what curated "if you loved X" lists are built on, but it gets
buried in the main TF-IDF bag, so we give it a dedicated channel here.

Backend: local sentence-transformer embeddings (``all-MiniLM-L6-v2``) when the optional
``sentence-transformers`` package is installed, cached to ``data/theme_embeddings.pkl`` and
invalidated by a content hash. When the package is absent we fall back to a TF-IDF model
over the same theme text, so the app and tests never hard-depend on the embedding library.

Both ``recommender.py`` and ``curator.py`` call into here so they share one definition of
"theme".
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EMBED_CACHE_PATH = Path("data/theme_embeddings.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"
SCORE_SCALE = 4.0          # match content_score / anchor_score range
SMALL_SET = 8              # below this many candidates, max-scale (keeps tiny unit tests monotonic)
NEG_PENALTY_SCALE = 1.5    # mirror add_content_similarity's negative penalty
SIM_PERCENTILE = 95        # robust reference instead of the single max (avoids outlier collapse)

_model = None
_model_loaded = False


def _as_list(value) -> List[str]:
    # Lazy import keeps this module free of a top-level recommender dependency
    # (recommender imports theme_similarity at module load).
    from recommender import _as_list as _al
    return _al(value)


def theme_text(row: pd.Series) -> str:
    """Concept text for a film: keywords + overview only. No genre/director/cast."""
    keywords = _as_list(row.get("keywords", []))
    overview = str(row.get("overview", "") or "").strip()
    parts: List[str] = []
    if keywords:
        parts.append(", ".join(keywords))
    if overview:
        parts.append(overview)
    return " \n ".join(parts).strip().lower()


def _text_by_id(meta: pd.DataFrame) -> Dict[str, str]:
    if meta is None or meta.empty or "movie_id" not in meta.columns:
        return {}
    return {str(r["movie_id"]): theme_text(r) for _, r in meta.iterrows()}


# --- embedding backend -------------------------------------------------------

def get_model():
    """Return a cached SentenceTransformer, or None if the library isn't installed."""
    global _model, _model_loaded
    if _model_loaded:
        return _model
    _model_loaded = True
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    except Exception:
        _model = None
    return _model


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_cache() -> Dict[str, Dict]:
    if EMBED_CACHE_PATH.exists():
        try:
            with EMBED_CACHE_PATH.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, Dict]) -> None:
    try:
        EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EMBED_CACHE_PATH.open("wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass


def embed_texts(text_by_id: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Embed each movie's theme text, reusing disk-cached vectors when the hash matches.

    Returns {} when no embedding model is available (callers then use the TF-IDF path).
    Vectors are L2-normalized so cosine similarity is a plain dot product.
    """
    model = get_model()
    if model is None:
        return {}
    cache = _load_cache()
    result: Dict[str, np.ndarray] = {}
    miss_ids: List[str] = []
    miss_texts: List[str] = []
    for mid, text in text_by_id.items():
        if not text:
            continue
        h = _hash(text)
        entry = cache.get(mid)
        if entry and entry.get("hash") == h:
            result[mid] = np.asarray(entry["vec"], dtype=np.float32)
        else:
            miss_ids.append(mid)
            miss_texts.append(text)
    if miss_texts:
        vecs = np.asarray(
            model.encode(miss_texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )
        for mid, text, vec in zip(miss_ids, miss_texts, vecs):
            result[mid] = vec
            cache[mid] = {"hash": _hash(text), "vec": vec}
        _save_cache(cache)
    return result


# --- scoring helpers ---------------------------------------------------------

def _scale_net(values: np.ndarray) -> np.ndarray:
    """Scale net similarities to roughly [-SCORE_SCALE, SCORE_SCALE].

    Uses a high percentile of the positive values as the reference rather than the single
    max, so one near-identical outlier (e.g. a same-director film) can't flatten genuine
    concept-cousins toward zero. Falls back to max-scaling for tiny candidate sets.
    """
    if values.size == 0:
        return values
    pos = values[values > 0]
    if values.size < SMALL_SET or pos.size == 0:
        ref = values.max()
    else:
        ref = float(np.percentile(pos, SIM_PERCENTILE))
    if ref <= 0:
        return np.zeros_like(values)
    return np.clip(values / ref, -1.0, 1.0) * SCORE_SCALE


def _embed_sim_matrix(cand_ids: List[str], group_ids: List[str], vecs: Dict[str, np.ndarray]) -> np.ndarray:
    """Cosine matrix (n_candidates x n_group) from normalized embeddings; 0 rows where no vec."""
    n, g = len(cand_ids), len(group_ids)
    matrix = np.zeros((n, g), dtype=np.float32)
    if g == 0:
        return matrix
    gmat = np.array([vecs[m] for m in group_ids], dtype=np.float32)  # (g, d)
    rows, idx = [], []
    for i, mid in enumerate(cand_ids):
        v = vecs.get(mid)
        if v is not None:
            rows.append(v)
            idx.append(i)
    if rows:
        sims = np.asarray(rows, dtype=np.float32) @ gmat.T  # (k, g) — cosine (normalized)
        for r, i in enumerate(idx):
            matrix[i] = sims[r]
    return matrix


def _tfidf_sim_matrix(cand_texts: List[str], group_texts: List[str]) -> np.ndarray:
    """Cosine matrix (n_candidates x n_group) via TF-IDF over theme text; 0 rows where empty."""
    n, g = len(cand_texts), len(group_texts)
    matrix = np.zeros((n, g), dtype=np.float32)
    if g == 0:
        return matrix
    nonempty = [i for i, t in enumerate(cand_texts) if t]
    if not nonempty:
        return matrix
    corpus = group_texts + [cand_texts[i] for i in nonempty]
    tfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=12000).fit_transform(corpus)
    gmat = tfidf[:g]
    cmat = tfidf[g:]
    sims = cosine_similarity(cmat, gmat)  # (len(nonempty), g)
    for r, i in enumerate(nonempty):
        matrix[i] = sims[r]
    return matrix


def _sim_matrix(cand_ids: List[str], cand_texts: List[str],
                group_ids: List[str], group_texts: List[str],
                vecs: Dict[str, np.ndarray]) -> np.ndarray:
    if vecs:
        return _embed_sim_matrix(cand_ids, group_ids, vecs)
    return _tfidf_sim_matrix(cand_texts, group_texts)


def theme_anchor_scores(candidates: pd.DataFrame, anchor_movie_id: str | None, meta: pd.DataFrame) -> pd.Series:
    """Theme similarity (0..4) of each candidate to one anchor film."""
    out = pd.Series(0.0, index=candidates.index)
    if not anchor_movie_id or meta is None or getattr(meta, "empty", True) or "movie_id" not in candidates.columns:
        return out
    text_by_id = _text_by_id(meta)
    anchor_text = text_by_id.get(str(anchor_movie_id), "")
    if not anchor_text:
        return out
    cand_ids = candidates["movie_id"].astype(str).tolist()
    cand_texts = [text_by_id.get(mid, "") for mid in cand_ids]
    vecs = embed_texts({str(anchor_movie_id): anchor_text,
                        **{mid: t for mid, t in zip(cand_ids, cand_texts) if t}})
    sims = _sim_matrix(cand_ids, cand_texts, [str(anchor_movie_id)], [anchor_text], vecs)
    out.iloc[:] = _scale_net(sims[:, 0])
    return out


def theme_taste_scores(candidates: pd.DataFrame, ratings: pd.DataFrame, likes: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
    """Theme similarity (≈ -4..4) of each candidate to the themes of your high-rated films.

    Mirrors add_content_similarity's positive/negative selection and rating weighting, but in
    theme-embedding (or theme-TF-IDF) space instead of the full feature bag.
    """
    out = pd.Series(0.0, index=candidates.index)
    if meta is None or getattr(meta, "empty", True) or "movie_id" not in candidates.columns:
        return out
    text_by_id = _text_by_id(meta)
    if not text_by_id:
        return out

    rated = ratings.copy() if ratings is not None else pd.DataFrame()
    if not rated.empty:
        rated["Rating"] = pd.to_numeric(rated.get("Rating"), errors="coerce")
    liked_ids = set(likes.get("movie_id", pd.Series(dtype=str)).dropna()) if likes is not None and not likes.empty else set()
    positive_ids = (set(rated.loc[rated["Rating"] >= 4.0, "movie_id"].dropna()) if not rated.empty else set()) | liked_ids
    negative_ids = set(rated.loc[rated["Rating"] <= 2.5, "movie_id"].dropna()) if not rated.empty else set()

    pos_ids = [mid for mid in positive_ids if text_by_id.get(str(mid))]
    neg_ids = [mid for mid in negative_ids if text_by_id.get(str(mid))]
    if not pos_ids:
        return out

    cand_ids = candidates["movie_id"].astype(str).tolist()
    cand_texts = [text_by_id.get(mid, "") for mid in cand_ids]
    pos_texts = [text_by_id[str(mid)] for mid in pos_ids]
    neg_texts = [text_by_id[str(mid)] for mid in neg_ids]

    vecs = embed_texts({
        **{mid: t for mid, t in zip(cand_ids, cand_texts) if t},
        **{str(mid): text_by_id[str(mid)] for mid in pos_ids},
        **{str(mid): text_by_id[str(mid)] for mid in neg_ids},
    })

    from recommender import _rating_weight
    rating_lookup = rated.dropna(subset=["Rating"]).set_index("movie_id")["Rating"] if not rated.empty else pd.Series(dtype=float)
    pos_weights = np.array([
        _rating_weight(rating_lookup[mid]) if mid in getattr(rating_lookup, "index", []) else _rating_weight(4.0)
        for mid in pos_ids
    ], dtype=np.float32)

    pos_sims = _sim_matrix(cand_ids, cand_texts, [str(m) for m in pos_ids], pos_texts, vecs)
    weight_sum = pos_weights.sum()
    weighted_pos = (pos_sims * pos_weights).sum(axis=1) / weight_sum if weight_sum > 0 else pos_sims.mean(axis=1)

    if neg_ids:
        neg_sims = _sim_matrix(cand_ids, cand_texts, [str(m) for m in neg_ids], neg_texts, vecs)
        neg_penalty = neg_sims.mean(axis=1) * NEG_PENALTY_SCALE
    else:
        neg_penalty = np.zeros(len(cand_ids), dtype=np.float32)

    out.iloc[:] = _scale_net(weighted_pos - neg_penalty)
    return out


def shared_theme_keywords(row_a: pd.Series, row_b: pd.Series, limit: int = 4) -> List[str]:
    """Literal keyword overlap, for human-readable 'shared themes' explanations."""
    a = {k.lower() for k in _as_list(row_a.get("keywords", []))}
    b = {k.lower() for k in _as_list(row_b.get("keywords", []))}
    return sorted(a & b)[:limit]
