from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import pandas as pd
import requests

SYNC_DIR = Path("data/sync")
EVENTS_PATH = SYNC_DIR / "rss_events.csv"
RATINGS_OVERLAY_PATH = SYNC_DIR / "ratings_overlay.csv"
WATCHED_OVERLAY_PATH = SYNC_DIR / "watched_overlay.csv"
DIARY_OVERLAY_PATH = SYNC_DIR / "diary_overlay.csv"
STATE_PATH = SYNC_DIR / "sync_state.json"

LB_NS = "https://letterboxd.com"
TMDB_NS = "https://www.themoviedb.org"


def username_to_rss_url(username_or_url: str) -> str:
    value = str(username_or_url).strip()
    if not value:
        raise ValueError("Letterboxd username or RSS URL is required.")
    if value.startswith("http://") or value.startswith("https://"):
        if value.endswith("/rss/") or value.endswith("/rss"):
            return value.rstrip("/") + "/"
        parsed = urlparse(value)
        parts = [p for p in parsed.path.split("/") if p]
        if not parts:
            raise ValueError("Could not infer username from URL.")
        return f"https://letterboxd.com/{parts[0]}/rss/"
    return f"https://letterboxd.com/{value.strip('/')}/rss/"


def _text_any(item: ET.Element, names: List[str]) -> str:
    for name in names:
        found = item.find(name)
        if found is not None and found.text:
            return found.text.strip()
    # Namespace-insensitive fallback.
    wanted = {n.split("}")[-1].split(":")[-1].lower() for n in names}
    for child in list(item):
        local = child.tag.split("}")[-1].split(":")[-1].lower()
        if local in wanted and child.text:
            return child.text.strip()
    return ""


def _parse_date(value: str) -> str:
    if not value:
        return ""
    value = value.strip()
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
            return value
        dt = parsedate_to_datetime(value)
        return dt.date().isoformat()
    except Exception:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
        except Exception:
            return ""


def _parse_rating(raw: str) -> float | None:
    if not raw:
        return None
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        pass
    stars = raw.count("★")
    half = 0.5 if "½" in raw or "1/2" in raw else 0.0
    if stars or half:
        return float(stars) + half
    match = re.search(r"([0-5](?:\.5)?)", raw)
    return float(match.group(1)) if match else None


def _title_year_from_title(title: str) -> Tuple[str, int | None]:
    # Common examples include: "Film Title, 1999 - ★★★½" or "Film Title, 1999"
    cleaned = re.sub(r"\s+-\s+[★½0-9. /]+.*$", "", title or "").strip()
    match = re.match(r"^(.*?),\s*(\d{4})$", cleaned)
    if match:
        return match.group(1).strip(), int(match.group(2))
    return cleaned.strip(), None


def _movie_id(name: str, year: Any) -> str:
    y = "<NA>" if year in (None, "") or pd.isna(year) else str(int(year))
    return f"{str(name).strip().lower()} ({y})"


def parse_rss_items(xml_text: str) -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    rows: List[Dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = _text_any(item, ["title"])
        link = _text_any(item, ["link"])
        guid = _text_any(item, ["guid"])
        pub_date = _parse_date(_text_any(item, ["pubDate"]))
        film_title = _text_any(item, [f"{{{LB_NS}}}filmTitle", "letterboxd:filmTitle", "filmTitle"])
        film_year_raw = _text_any(item, [f"{{{LB_NS}}}filmYear", "letterboxd:filmYear", "filmYear"])
        if not film_title:
            film_title, fallback_year = _title_year_from_title(title)
        else:
            fallback_year = None
        try:
            film_year = int(film_year_raw) if film_year_raw else fallback_year
        except ValueError:
            film_year = fallback_year
        watched_date = _parse_date(_text_any(item, [f"{{{LB_NS}}}watchedDate", "letterboxd:watchedDate", "watchedDate"])) or pub_date
        rating_raw = _text_any(item, [f"{{{LB_NS}}}memberRating", "letterboxd:memberRating", "memberRating"])
        rating = _parse_rating(rating_raw or title)
        rewatch_raw = _text_any(item, [f"{{{LB_NS}}}rewatch", "letterboxd:rewatch", "rewatch"]).lower()
        rewatch = rewatch_raw in {"yes", "true", "1"}
        tmdb_id = _text_any(item, [f"{{{TMDB_NS}}}movieId", "tmdb:movieId", "movieId"])
        if not film_title:
            continue
        rows.append({
            "event_id": guid or link or f"{film_title}|{film_year}|{watched_date}|{rating}",
            "Name": film_title,
            "Year": film_year,
            "movie_id": _movie_id(film_title, film_year),
            "Rating": rating,
            "Watched Date": watched_date,
            "Rewatch": rewatch,
            "Letterboxd URI": link,
            "tmdb_id": int(tmdb_id) if str(tmdb_id).isdigit() else pd.NA,
            "rss_title": title,
            "rss_published": pub_date,
            "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    return pd.DataFrame(rows)


def fetch_rss(username_or_url: str, timeout: int = 20) -> pd.DataFrame:
    url = username_to_rss_url(username_or_url)
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "movie-recommender-mvp/1.0"})
    response.raise_for_status()
    df = parse_rss_items(response.text)
    if not df.empty:
        df["rss_url"] = url
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def merge_rss_events(new_events: pd.DataFrame, sync_dir: str | Path = SYNC_DIR) -> Dict[str, Any]:
    sync_dir = Path(sync_dir)
    events_path = sync_dir / "rss_events.csv"
    ratings_path = sync_dir / "ratings_overlay.csv"
    watched_path = sync_dir / "watched_overlay.csv"
    diary_path = sync_dir / "diary_overlay.csv"
    state_path = sync_dir / "sync_state.json"
    if new_events.empty:
        return {"new_events": 0, "total_events": len(_read_csv(events_path)) if events_path.exists() else 0}

    existing = _read_csv(events_path)
    combined = pd.concat([existing, new_events], ignore_index=True, sort=False) if not existing.empty else new_events.copy()
    combined = combined.drop_duplicates("event_id", keep="last")
    _write_csv(combined, events_path)

    # Watched overlay: one row per watched film.
    watched = combined[["Name", "Year", "Letterboxd URI", "movie_id"]].drop_duplicates("movie_id", keep="last")
    _write_csv(watched, watched_path)

    # Ratings overlay: latest known rating per movie. This captures changed ratings when the film appears again in RSS.
    rated = combined.dropna(subset=["Rating"]).copy()
    if not rated.empty:
        rated = rated.sort_values(["rss_published", "synced_at"]).drop_duplicates("movie_id", keep="last")
        ratings = rated[["Name", "Year", "Rating", "Letterboxd URI", "movie_id", "synced_at"]]
    else:
        ratings = pd.DataFrame(columns=["Name", "Year", "Rating", "Letterboxd URI", "movie_id", "synced_at"])
    _write_csv(ratings, ratings_path)

    diary_cols = ["Name", "Year", "Watched Date", "Rewatch", "Rating", "Letterboxd URI", "movie_id", "event_id", "synced_at"]
    diary = combined[[c for c in diary_cols if c in combined.columns]].drop_duplicates("event_id", keep="last")
    _write_csv(diary, diary_path)

    state = {
        "last_sync_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rss_url": str(new_events.get("rss_url", pd.Series([""])).dropna().iloc[0]) if "rss_url" in new_events.columns and not new_events.empty else "",
        "total_events": int(len(combined)),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return {"new_events": int(len(combined) - len(existing.drop_duplicates("event_id")) if not existing.empty else len(new_events.drop_duplicates("event_id"))), "total_events": int(len(combined))}


def sync_rss(username_or_url: str, sync_dir: str | Path = SYNC_DIR) -> Dict[str, Any]:
    events = fetch_rss(username_or_url)
    result = merge_rss_events(events, sync_dir=sync_dir)
    result["fetched_events"] = int(len(events))
    return result


def sync_status(sync_dir: str | Path = SYNC_DIR) -> Dict[str, Any]:
    sync_dir = Path(sync_dir)
    state_path = sync_dir / "sync_state.json"
    state: Dict[str, Any] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    for name, path in {
        "rss_events": sync_dir / "rss_events.csv",
        "ratings_overlay": sync_dir / "ratings_overlay.csv",
        "watched_overlay": sync_dir / "watched_overlay.csv",
        "diary_overlay": sync_dir / "diary_overlay.csv",
    }.items():
        state[name] = len(_read_csv(path)) if path.exists() else 0
    return state


def apply_sync_overlays(data: Dict[str, pd.DataFrame], sync_dir: str | Path = SYNC_DIR) -> Dict[str, pd.DataFrame]:
    from recommender import normalize_movie_key

    sync_dir = Path(sync_dir)
    out = {k: v.copy() for k, v in data.items()}

    watched_overlay = _read_csv(sync_dir / "watched_overlay.csv")
    if not watched_overlay.empty:
        watched_overlay = normalize_movie_key(watched_overlay)
        out["watched"] = pd.concat([out["watched"], watched_overlay], ignore_index=True, sort=False).drop_duplicates("movie_id", keep="last")

    diary_overlay = _read_csv(sync_dir / "diary_overlay.csv")
    if not diary_overlay.empty:
        diary_overlay = normalize_movie_key(diary_overlay)
        out["diary"] = pd.concat([out["diary"], diary_overlay], ignore_index=True, sort=False).drop_duplicates(["movie_id", "Watched Date"], keep="last")

    ratings_overlay = _read_csv(sync_dir / "ratings_overlay.csv")
    if not ratings_overlay.empty:
        ratings_overlay = normalize_movie_key(ratings_overlay)
        base = out["ratings"].copy()
        if "Rating" in base.columns:
            base = base[~base["movie_id"].isin(set(ratings_overlay["movie_id"]))]
        out["ratings"] = pd.concat([base, ratings_overlay], ignore_index=True, sort=False).drop_duplicates("movie_id", keep="last")

    return out
