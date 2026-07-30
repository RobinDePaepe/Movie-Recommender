from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

# Letterboxd has no public API for average rating; the page's twitter:data2 meta tag
# ("4.52 out of 5") is the same value the site itself renders, so we parse that.
RATING_RE = re.compile(r'twitter:data2["\']\s*content=["\']([\d.]+) out of 5["\']')
USER_AGENT = "Mozilla/5.0 (compatible; personal-movie-recommender/1.0)"

# Personal log/diary URIs (from RSS sync, e.g. letterboxd.com/{username}/film/{slug}/) carry
# *your* star rating in twitter:data2, not the film's average — normalize to the canonical
# /film/{slug}/ page before fetching. boxd.it short links already redirect there untouched.
_PERSONAL_URI_RE = re.compile(r"^(https?://letterboxd\.com/)[^/]+/(film/.+)$")


def _canonical_film_uri(uri: str) -> str:
    match = _PERSONAL_URI_RE.match(uri)
    return match.group(1) + match.group(2) if match else uri


@dataclass
class LetterboxdRatingClient:
    """Scrapes each film's average Letterboxd rating, keyed and cached by the film's Letterboxd URI."""

    cache_path: Path = Path("data/letterboxd_rating_cache.json")
    sleep_seconds: float = 1.0

    def __post_init__(self) -> None:
        self.cache_path = Path(self.cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            backup = self.cache_path.with_suffix(".broken.json")
            self.cache_path.rename(backup)
            return {}

    def save(self) -> None:
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        try:
            tmp.replace(self.cache_path)
        except PermissionError:
            self.cache_path.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")
            try:
                tmp.unlink()
            except OSError:
                pass

    def fetch_rating(self, letterboxd_uri: str, force: bool = False) -> Dict[str, Any]:
        if not letterboxd_uri:
            return {}
        if letterboxd_uri in self.cache and not force:
            return self.cache[letterboxd_uri]
        record: Dict[str, Any] = {"letterboxd_uri": letterboxd_uri, "letterboxd_rating": None}
        try:
            fetch_uri = _canonical_film_uri(letterboxd_uri)
            response = requests.get(fetch_uri, timeout=20, headers={"User-Agent": USER_AGENT}, allow_redirects=True)
            response.raise_for_status()
            time.sleep(self.sleep_seconds)
            match = RATING_RE.search(response.text)
            record["letterboxd_rating"] = float(match.group(1)) if match else None
            if not match:
                record["error"] = "Rating not found on page"
        except Exception as exc:
            record["error"] = str(exc)
        self.cache[letterboxd_uri] = record
        self.save()
        return record


def enrich_letterboxd_ratings(movies: pd.DataFrame, client: LetterboxdRatingClient, limit: Optional[int] = None, force: bool = False) -> Dict[str, float]:
    """Scrape average Letterboxd ratings for films with a known Letterboxd URI.

    `movies` needs `movie_id` and `Letterboxd URI` columns. Returns {movie_id: letterboxd_rating}.
    """
    results: Dict[str, float] = {}
    if movies.empty or "Letterboxd URI" not in movies.columns:
        return results
    candidates = movies.dropna(subset=["Letterboxd URI"]).drop_duplicates("movie_id")
    candidates = candidates[candidates["Letterboxd URI"].astype(str).str.strip() != ""]
    fetched = 0
    for _, row in candidates.iterrows():
        uri = str(row["Letterboxd URI"]).strip()
        if not force and uri in client.cache:
            cached = client.cache[uri]
            if cached.get("letterboxd_rating") is not None:
                results[row["movie_id"]] = cached["letterboxd_rating"]
            continue
        record = client.fetch_rating(uri, force=force)
        if record.get("letterboxd_rating") is not None:
            results[row["movie_id"]] = record["letterboxd_rating"]
        fetched += 1
        if limit is not None and fetched >= limit:
            break
    return results
