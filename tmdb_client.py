from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w342"


@dataclass
class TMDbClient:
    api_key: str
    cache_path: Path = Path("data/tmdb_cache.json")
    sleep_seconds: float = 0.25

    def __post_init__(self) -> None:
        self.cache_path = Path(self.cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()

    @classmethod
    def from_env(cls, cache_path: str | Path = "data/tmdb_cache.json") -> Optional["TMDbClient"]:
        api_key = os.getenv("TMDB_API_KEY") or os.getenv("TMDB_KEY")
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

    @staticmethod
    def cache_key(name: str, year: Any) -> str:
        y = "" if pd.isna(year) else str(int(year))
        return f"{str(name).strip().lower()}|{y}"

    @staticmethod
    def tmdb_key(tmdb_id: int | str) -> str:
        return f"tmdb:{int(tmdb_id)}"

    def _get(self, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        params = {**(params or {}), "api_key": self.api_key}
        response = requests.get(f"{TMDB_BASE_URL}{endpoint}", params=params, timeout=20)
        response.raise_for_status()
        time.sleep(self.sleep_seconds)
        return response.json()

    def search_movie(self, name: str, year: Any) -> Optional[Dict[str, Any]]:
        params: Dict[str, Any] = {"query": name, "include_adult": "false"}
        if not pd.isna(year):
            params["year"] = int(year)
        payload = self._get("/search/movie", params)
        results = payload.get("results", [])
        if not results and not pd.isna(year):
            payload = self._get("/search/movie", {"query": name, "include_adult": "false"})
            results = payload.get("results", [])
        return results[0] if results else None

    def movie_details(self, tmdb_id: int) -> Dict[str, Any]:
        return self._get(f"/movie/{tmdb_id}", {"append_to_response": "credits,keywords"})

    def similar_movies(self, tmdb_id: int, limit: int = 20) -> list[Dict[str, Any]]:
        payload = self._get(f"/movie/{tmdb_id}/similar", {"page": 1})
        return payload.get("results", [])[:limit]

    def recommended_movies(self, tmdb_id: int, limit: int = 20) -> list[Dict[str, Any]]:
        payload = self._get(f"/movie/{tmdb_id}/recommendations", {"page": 1})
        return payload.get("results", [])[:limit]

    def fetch_movie_metadata(self, name: str, year: Any, force: bool = False) -> Dict[str, Any]:
        key = self.cache_key(name, year)
        if key in self.cache and not force:
            entry = self.cache[key]
            if isinstance(entry, str) and entry.startswith("tmdb:"):
                return self.cache.get(entry, {})
            return entry

        record: Dict[str, Any] = {"name": name, "year": None if pd.isna(year) else int(year), "tmdb_found": False}
        try:
            match = self.search_movie(name, year)
            if not match:
                record["error"] = "No TMDb match found"
            else:
                record = self.fetch_movie_metadata_by_tmdb_id(int(match["id"]), source="letterboxd", force=force)
                self.cache[key] = self.tmdb_key(record["tmdb_id"])
                self.save()
                return record
        except Exception as exc:
            record["error"] = str(exc)

        self.cache[key] = record
        self.save()
        return record

    def fetch_movie_metadata_by_tmdb_id(self, tmdb_id: int, source: str = "tmdb", force: bool = False) -> Dict[str, Any]:
        tid_key = self.tmdb_key(tmdb_id)
        if tid_key in self.cache and not force:
            return self.cache[tid_key]
        record: Dict[str, Any] = {"name": "", "year": None, "tmdb_found": False, "source": source}
        try:
            details = self.movie_details(int(tmdb_id))
            record.update(flatten_tmdb_details(details))
            record["tmdb_found"] = True
            record["source"] = source
            name_key = self.cache_key(record["name"], record["year"])
            self.cache[tid_key] = record
            self.cache[name_key] = tid_key
        except Exception as exc:
            record["tmdb_id"] = tmdb_id
            record["error"] = str(exc)
            self.cache[tid_key] = record
        self.save()
        return record

    def discover_from_seed(self, tmdb_id: int, limit: int = 10, force: bool = False) -> list[Dict[str, Any]]:
        candidates: list[Dict[str, Any]] = []
        seen: set[int] = set()
        for source_name, fn in [("tmdb_recommendations", self.recommended_movies), ("tmdb_similar", self.similar_movies)]:
            for item in fn(int(tmdb_id), limit=limit):
                tid = item.get("id")
                if not tid or int(tid) in seen:
                    continue
                seen.add(int(tid))
                rec = self.fetch_movie_metadata_by_tmdb_id(int(tid), source=source_name, force=force)
                candidates.append(rec)
        return candidates


def _release_year(details: Dict[str, Any]) -> int | None:
    date = details.get("release_date") or ""
    if len(date) >= 4 and date[:4].isdigit():
        return int(date[:4])
    return None


def flatten_tmdb_details(details: Dict[str, Any]) -> Dict[str, Any]:
    credits = details.get("credits", {}) or {}
    crew = credits.get("crew", []) or []
    cast = credits.get("cast", []) or []
    keywords_payload = details.get("keywords", {}) or {}

    directors = [p.get("name") for p in crew if p.get("job") == "Director" and p.get("name")]
    writers = [p.get("name") for p in crew if p.get("job") in {"Writer", "Screenplay", "Story"} and p.get("name")]
    top_cast = [p.get("name") for p in cast[:10] if p.get("name")]
    genres = [g.get("name") for g in details.get("genres", []) if g.get("name")]
    countries = [c.get("name") for c in details.get("production_countries", []) if c.get("name")]
    languages = [l.get("english_name") for l in details.get("spoken_languages", []) if l.get("english_name")]
    keywords = [k.get("name") for k in keywords_payload.get("keywords", []) if k.get("name")]
    poster_path = details.get("poster_path")

    return {
        "name": details.get("title") or details.get("name") or "",
        "year": _release_year(details),
        "tmdb_id": details.get("id"),
        "tmdb_title": details.get("title"),
        "tmdb_release_date": details.get("release_date"),
        "overview": details.get("overview") or "",
        "genres": genres,
        "directors": directors,
        "writers": sorted(set(writers)),
        "cast": top_cast,
        "keywords": keywords,
        "countries": countries,
        "languages": languages,
        "runtime": details.get("runtime"),
        "tmdb_vote_average": details.get("vote_average"),
        "tmdb_vote_count": details.get("vote_count"),
        "tmdb_popularity": details.get("popularity"),
        "poster_path": poster_path,
        "poster_url": f"{POSTER_BASE_URL}{poster_path}" if poster_path else "",
        "tmdb_url": f"https://www.themoviedb.org/movie/{details.get('id')}",
    }


def enrich_movies(movies: pd.DataFrame, client: TMDbClient, limit: int | None = None, force: bool = False) -> pd.DataFrame:
    rows = []
    fetched = 0
    frame = movies[["Name", "Year"]].drop_duplicates().copy()
    for _, row in frame.iterrows():
        key = client.cache_key(row["Name"], row["Year"])
        if not force and key in client.cache:
            continue
        rows.append(client.fetch_movie_metadata(row["Name"], row["Year"], force=force))
        fetched += 1
        if limit is not None and fetched >= limit:
            break
    return pd.DataFrame(rows)


def discover_movies_from_favorites(
    favorite_movies: pd.DataFrame,
    client: TMDbClient,
    per_seed: int = 8,
    seed_limit: int = 25,
    force: bool = False,
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    seeds = favorite_movies.dropna(subset=["tmdb_id"]).drop_duplicates("tmdb_id").head(seed_limit)
    for _, seed in seeds.iterrows():
        for record in client.discover_from_seed(int(seed["tmdb_id"]), limit=per_seed, force=force):
            record = dict(record)
            record["discovered_from"] = f"{seed.get('Name') or seed.get('name')} ({seed.get('Year') or seed.get('year')})"
            rows.append(record)
    return pd.DataFrame(rows).drop_duplicates("tmdb_id") if rows else pd.DataFrame()


def _resolve_cache_entry(cache: Dict[str, Any], entry: Any) -> Dict[str, Any]:
    if isinstance(entry, str) and entry.startswith("tmdb:"):
        entry = cache.get(entry, {})
    return entry if isinstance(entry, dict) else {}


def metadata_from_cache(movies: pd.DataFrame | None = None, cache_path: str | Path = "data/tmdb_cache.json", include_all: bool = False) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return pd.DataFrame()
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    rows = []
    seen_keys: set[str] = set()

    if movies is not None and not movies.empty and not include_all:
        for _, row in movies[["Name", "Year"]].drop_duplicates().iterrows():
            key = TMDbClient.cache_key(row["Name"], row["Year"])
            entry = _resolve_cache_entry(cache, cache.get(key))
            if entry:
                dedupe = str(entry.get("tmdb_id") or key)
                if dedupe not in seen_keys:
                    rows.append(entry)
                    seen_keys.add(dedupe)
    else:
        for key, entry in cache.items():
            entry = _resolve_cache_entry(cache, entry)
            if not entry:
                continue
            dedupe = str(entry.get("tmdb_id") or key)
            if dedupe in seen_keys:
                continue
            rows.append(entry)
            seen_keys.add(dedupe)
    return pd.DataFrame(rows)
