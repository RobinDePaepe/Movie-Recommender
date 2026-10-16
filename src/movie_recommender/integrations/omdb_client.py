from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

OMDB_BASE_URL = "https://www.omdbapi.com/"


@dataclass
class OMDbClient:
    """Fetches IMDb ratings (via OMDb, since TMDb doesn't expose the real IMDb score) by imdb_id."""

    api_key: str
    cache_path: Path = Path("data/omdb_cache.json")
    sleep_seconds: float = 0.25

    def __post_init__(self) -> None:
        self.cache_path = Path(self.cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()

    @classmethod
    def from_env(cls, cache_path: str | Path = "data/omdb_cache.json") -> Optional["OMDbClient"]:
        api_key = os.getenv("OMDB_API_KEY")
        if not api_key:
            return None
        return cls(api_key=api_key, cache_path=Path(cache_path))

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

    def fetch_by_imdb_id(self, imdb_id: str, force: bool = False) -> Dict[str, Any]:
        if not imdb_id:
            return {}
        if imdb_id in self.cache and not force:
            return self.cache[imdb_id]
        record: Dict[str, Any] = {"imdb_id": imdb_id, "imdb_rating": None, "imdb_votes": None}
        try:
            response = requests.get(OMDB_BASE_URL, params={"i": imdb_id, "apikey": self.api_key}, timeout=20)
            response.raise_for_status()
            payload = response.json()
            time.sleep(self.sleep_seconds)
            if payload.get("Response") == "True":
                rating = payload.get("imdbRating")
                votes = payload.get("imdbVotes")
                record["imdb_rating"] = float(rating) if rating and rating != "N/A" else None
                record["imdb_votes"] = int(str(votes).replace(",", "")) if votes and votes != "N/A" else None
            else:
                record["error"] = payload.get("Error", "Unknown OMDb error")
        except Exception as exc:
            record["error"] = str(exc)
        self.cache[imdb_id] = record
        self.save()
        return record


def enrich_imdb_ratings(metadata: pd.DataFrame, client: OMDbClient, limit: Optional[int] = None, force: bool = False) -> Dict[str, Dict[str, Any]]:
    """Fetch IMDb ratings for films that already have an imdb_id (from TMDb enrichment).

    Returns {movie_id: {"imdb_rating": float|None, "imdb_votes": int|None}} for movies fetched this run.
    """
    results: Dict[str, Dict[str, Any]] = {}
    if metadata.empty or "imdb_id" not in metadata.columns:
        return results
    candidates = metadata.dropna(subset=["imdb_id"]).drop_duplicates("movie_id")
    fetched = 0
    for _, row in candidates.iterrows():
        imdb_id = str(row["imdb_id"])
        if not force and imdb_id in client.cache:
            cached = client.cache[imdb_id]
            if cached.get("imdb_rating") is not None:
                results[row["movie_id"]] = {"imdb_rating": cached.get("imdb_rating"), "imdb_votes": cached.get("imdb_votes")}
            continue
        record = client.fetch_by_imdb_id(imdb_id, force=force)
        results[row["movie_id"]] = {"imdb_rating": record.get("imdb_rating"), "imdb_votes": record.get("imdb_votes")}
        fetched += 1
        if limit is not None and fetched >= limit:
            break
    return results
