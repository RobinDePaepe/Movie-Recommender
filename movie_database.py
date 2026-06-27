from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from recommender import ensure_export_dir, load_letterboxd, normalize_movie_key, load_feedback
from tmdb_client import metadata_from_cache

DB_PATH = Path("data/movie_recommender.sqlite")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: str | Path = DB_PATH):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | Path = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS movies (
                movie_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                year INTEGER,
                letterboxd_uri TEXT,
                tmdb_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ratings (
                movie_id TEXT PRIMARY KEY REFERENCES movies(movie_id) ON DELETE CASCADE,
                rating REAL,
                rated_at TEXT,
                source TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rating_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie_id TEXT REFERENCES movies(movie_id) ON DELETE CASCADE,
                old_rating REAL,
                new_rating REAL,
                changed_at TEXT NOT NULL,
                source TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS watched_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie_id TEXT REFERENCES movies(movie_id) ON DELETE CASCADE,
                watched_date TEXT,
                rewatch INTEGER DEFAULT 0,
                rating REAL,
                review_text TEXT,
                source TEXT NOT NULL,
                source_event_id TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(movie_id, watched_date, source_event_id)
            );

            CREATE TABLE IF NOT EXISTS watchlist (
                movie_id TEXT PRIMARY KEY REFERENCES movies(movie_id) ON DELETE CASCADE,
                added_at TEXT,
                removed_at TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                source TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS likes (
                movie_id TEXT PRIMARY KEY REFERENCES movies(movie_id) ON DELETE CASCADE,
                liked_at TEXT,
                source TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS list_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                list_name TEXT NOT NULL,
                movie_id TEXT REFERENCES movies(movie_id) ON DELETE CASCADE,
                position INTEGER,
                source TEXT NOT NULL,
                UNIQUE(list_name, movie_id)
            );

            CREATE TABLE IF NOT EXISTS movie_metadata (
                movie_id TEXT PRIMARY KEY REFERENCES movies(movie_id) ON DELETE CASCADE,
                tmdb_id INTEGER,
                tmdb_found INTEGER,
                tmdb_title TEXT,
                tmdb_release_date TEXT,
                overview TEXT,
                genres TEXT,
                directors TEXT,
                writers TEXT,
                cast TEXT,
                keywords TEXT,
                countries TEXT,
                languages TEXT,
                moods TEXT,
                runtime INTEGER,
                tmdb_vote_average REAL,
                tmdb_vote_count INTEGER,
                tmdb_popularity REAL,
                poster_path TEXT,
                poster_url TEXT,
                tmdb_url TEXT,
                discovered_from TEXT,
                raw_json TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie_id TEXT REFERENCES movies(movie_id) ON DELETE CASCADE,
                feedback TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sync_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                message TEXT
            );

            CREATE TABLE IF NOT EXISTS curated_weeks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anchor_movie_id TEXT NOT NULL,
                anchor_name TEXT NOT NULL,
                style TEXT NOT NULL,
                total_movies INTEGER NOT NULL,
                label TEXT,
                movies_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_watched_movie ON watched_events(movie_id);
            CREATE INDEX IF NOT EXISTS idx_ratings_rating ON ratings(rating);
            CREATE INDEX IF NOT EXISTS idx_metadata_tmdb_id ON movie_metadata(tmdb_id);
            """
        )


def _safe_year(year: Any) -> Optional[int]:
    try:
        if pd.isna(year):
            return None
        return int(year)
    except Exception:
        return None


def movie_id(name: str, year: Any) -> str:
    y = "<NA>" if pd.isna(year) else str(int(year))
    return f"{str(name).strip().lower()} ({y})"


def upsert_movie(conn: sqlite3.Connection, name: str, year: Any, uri: str = "", tmdb_id: Any = None) -> str:
    mid = movie_id(name, year)
    now = utc_now()
    conn.execute(
        """
        INSERT INTO movies(movie_id, name, year, letterboxd_uri, tmdb_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(movie_id) DO UPDATE SET
          name=excluded.name,
          year=excluded.year,
          letterboxd_uri=COALESCE(NULLIF(excluded.letterboxd_uri,''), movies.letterboxd_uri),
          tmdb_id=COALESCE(excluded.tmdb_id, movies.tmdb_id),
          updated_at=excluded.updated_at
        """,
        (mid, str(name).strip(), _safe_year(year), uri or "", None if pd.isna(tmdb_id) else tmdb_id, now, now),
    )
    return mid


def import_letterboxd_export(export_zip: str | Path = "data/letterboxd_export.zip", db_path: str | Path = DB_PATH, export_dir: str | Path = "data/letterboxd") -> Dict[str, int]:
    init_db(db_path)
    data = load_letterboxd(ensure_export_dir(export_zip, export_dir))
    counts = {"movies": 0, "ratings": 0, "watched_events": 0, "watchlist": 0, "likes": 0, "list_entries": 0}
    with connect(db_path) as conn:
        run_id = _start_run(conn, "letterboxd_export")
        try:
            for key in ["ratings", "watched", "watchlist", "likes"]:
                frame = data.get(key, pd.DataFrame())
                for _, row in frame.iterrows():
                    uri = row.get("Letterboxd URI", row.get("URL", ""))
                    upsert_movie(conn, row["Name"], row["Year"], uri=uri)
                    counts["movies"] += 1

            ratings = data.get("ratings", pd.DataFrame()).copy()
            if not ratings.empty:
                for _, row in ratings.iterrows():
                    mid = upsert_movie(conn, row["Name"], row["Year"], uri=row.get("Letterboxd URI", ""))
                    new_rating = pd.to_numeric(row.get("Rating"), errors="coerce")
                    old = conn.execute("SELECT rating FROM ratings WHERE movie_id=?", (mid,)).fetchone()
                    if old is not None and old["rating"] != float(new_rating):
                        conn.execute("INSERT INTO rating_history(movie_id, old_rating, new_rating, changed_at, source) VALUES (?, ?, ?, ?, ?)", (mid, old["rating"], float(new_rating), utc_now(), "letterboxd_export"))
                    conn.execute("INSERT INTO ratings(movie_id, rating, rated_at, source, updated_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(movie_id) DO UPDATE SET rating=excluded.rating, source=excluded.source, updated_at=excluded.updated_at", (mid, None if pd.isna(new_rating) else float(new_rating), row.get("Date", ""), "letterboxd_export", utc_now()))
                    counts["ratings"] += 1

            watched = data.get("watched", pd.DataFrame()).copy()
            if not watched.empty:
                for _, row in watched.iterrows():
                    mid = upsert_movie(conn, row["Name"], row["Year"], uri=row.get("Letterboxd URI", ""))
                    conn.execute("INSERT OR IGNORE INTO watched_events(movie_id, watched_date, rewatch, rating, review_text, source, source_event_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (mid, row.get("Date", ""), 0, None, "", "letterboxd_export", f"watched:{mid}:{row.get('Date','')}", utc_now()))
                    counts["watched_events"] += 1

            diary = data.get("diary", pd.DataFrame()).copy()
            if not diary.empty:
                for _, row in diary.iterrows():
                    mid = upsert_movie(conn, row["Name"], row["Year"], uri=row.get("Letterboxd URI", ""))
                    rating = pd.to_numeric(row.get("Rating"), errors="coerce")
                    rewatch = str(row.get("Rewatch", "")).lower() in {"yes", "true", "1"}
                    conn.execute("INSERT OR IGNORE INTO watched_events(movie_id, watched_date, rewatch, rating, review_text, source, source_event_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (mid, row.get("Watched Date", row.get("Date", "")), int(rewatch), None if pd.isna(rating) else float(rating), row.get("Review", ""), "letterboxd_diary", f"diary:{mid}:{row.get('Watched Date', row.get('Date',''))}:{row.get('Rating','')}", utc_now()))
                    counts["watched_events"] += 1

            watchlist = data.get("watchlist", pd.DataFrame()).copy()
            if not watchlist.empty:
                for _, row in watchlist.iterrows():
                    mid = upsert_movie(conn, row["Name"], row["Year"], uri=row.get("Letterboxd URI", ""))
                    conn.execute("INSERT INTO watchlist(movie_id, added_at, removed_at, active, source, updated_at) VALUES (?, ?, NULL, 1, ?, ?) ON CONFLICT(movie_id) DO UPDATE SET active=1, removed_at=NULL, source=excluded.source, updated_at=excluded.updated_at", (mid, row.get("Date", ""), "letterboxd_export", utc_now()))
                    counts["watchlist"] += 1

            likes = data.get("likes", pd.DataFrame()).copy()
            if not likes.empty:
                for _, row in likes.iterrows():
                    mid = upsert_movie(conn, row["Name"], row["Year"], uri=row.get("Letterboxd URI", ""))
                    conn.execute("INSERT INTO likes(movie_id, liked_at, source, updated_at) VALUES (?, ?, ?, ?) ON CONFLICT(movie_id) DO UPDATE SET liked_at=excluded.liked_at, source=excluded.source, updated_at=excluded.updated_at", (mid, row.get("Date", ""), "letterboxd_export", utc_now()))
                    counts["likes"] += 1

            lists = data.get("lists", pd.DataFrame()).copy()
            if not lists.empty:
                for _, row in lists.iterrows():
                    mid = upsert_movie(conn, row["Name"], row["Year"], uri=row.get("URL", ""))
                    conn.execute("INSERT OR IGNORE INTO list_entries(list_name, movie_id, position, source) VALUES (?, ?, ?, ?)", (row.get("source_list", ""), mid, None if pd.isna(row.get("Position")) else int(row.get("Position")), "letterboxd_export"))
                    counts["list_entries"] += 1
            _finish_run(conn, run_id, "ok", json.dumps(counts))
        except Exception as exc:
            _finish_run(conn, run_id, "error", str(exc))
            raise
    return counts


def import_tmdb_cache(cache_path: str | Path = "data/tmdb_cache.json", db_path: str | Path = DB_PATH) -> int:
    init_db(db_path)
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return 0
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    imported = 0
    with connect(db_path) as conn:
        for entry in cache.values():
            if isinstance(entry, str):
                continue
            name = entry.get("name") or entry.get("tmdb_title")
            year = entry.get("year")
            if not name:
                continue
            mid = upsert_movie(conn, name, year, tmdb_id=entry.get("tmdb_id"))
            _upsert_metadata(conn, mid, entry)
            imported += 1
    return imported


def apply_rss_overlays_to_db(sync_dir: str | Path = "data/sync", db_path: str | Path = DB_PATH) -> Dict[str, int]:
    """Upsert RSS overlay CSVs into the SQLite database.

    Applies ratings_overlay.csv (with rating-history tracking) and diary_overlay.csv
    (individual watch events, deduped by event_id).  Call this after sync_rss() so that
    SQLite-mode data stays in sync without a full rebuild.
    """
    sync_dir = Path(sync_dir)
    counts: Dict[str, int] = {"ratings_updated": 0, "diary_added": 0}

    if not sync_dir.exists():
        return counts

    def _read(name: str) -> pd.DataFrame:
        p = sync_dir / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    init_db(db_path)
    with connect(db_path) as conn:
        run_id = _start_run(conn, "rss_overlay")
        try:
            # Ratings overlay — upsert latest rating per movie, track changes.
            for _, row in _read("ratings_overlay.csv").iterrows():
                name = str(row.get("Name", "")).strip()
                if not name:
                    continue
                mid = upsert_movie(conn, name, row.get("Year"), uri=str(row.get("Letterboxd URI", "")))
                new_rating = pd.to_numeric(row.get("Rating"), errors="coerce")
                if pd.isna(new_rating):
                    continue
                old = conn.execute("SELECT rating FROM ratings WHERE movie_id=?", (mid,)).fetchone()
                if old is not None and old["rating"] != float(new_rating):
                    conn.execute(
                        "INSERT INTO rating_history(movie_id, old_rating, new_rating, changed_at, source) VALUES (?, ?, ?, ?, ?)",
                        (mid, old["rating"], float(new_rating), utc_now(), "letterboxd_rss"),
                    )
                conn.execute(
                    "INSERT INTO ratings(movie_id, rating, rated_at, source, updated_at) VALUES (?, ?, ?, ?, ?)"
                    " ON CONFLICT(movie_id) DO UPDATE SET"
                    "   rating=excluded.rating, source=excluded.source, updated_at=excluded.updated_at",
                    (mid, float(new_rating), str(row.get("synced_at", "")), "letterboxd_rss", utc_now()),
                )
                counts["ratings_updated"] += 1

            # Diary overlay — insert individual watch events; event_id prevents duplicates.
            for _, row in _read("diary_overlay.csv").iterrows():
                name = str(row.get("Name", "")).strip()
                if not name:
                    continue
                mid = upsert_movie(conn, name, row.get("Year"), uri=str(row.get("Letterboxd URI", "")))
                rating = pd.to_numeric(row.get("Rating"), errors="coerce")
                rewatch = str(row.get("Rewatch", "")).lower() in {"yes", "true", "1"}
                event_id = str(row.get("event_id", f"rss:{mid}:{row.get('Watched Date', '')}"))
                conn.execute(
                    "INSERT OR IGNORE INTO watched_events"
                    "(movie_id, watched_date, rewatch, rating, review_text, source, source_event_id, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        mid, str(row.get("Watched Date", "")), int(rewatch),
                        None if pd.isna(rating) else float(rating),
                        "", "letterboxd_rss", event_id, utc_now(),
                    ),
                )
                counts["diary_added"] += 1

            _finish_run(conn, run_id, "ok", json.dumps(counts))
        except Exception as exc:
            _finish_run(conn, run_id, "error", str(exc))
            raise

    return counts


def import_feedback_csv(path: str | Path = "data/feedback.csv", db_path: str | Path = DB_PATH) -> int:
    init_db(db_path)
    fb = load_feedback(path)
    if fb.empty:
        return 0
    count = 0
    with connect(db_path) as conn:
        for _, row in fb.iterrows():
            if not row.get("movie_id"):
                continue
            # Ensure placeholder movie exists if needed.
            existing = conn.execute("SELECT movie_id FROM movies WHERE movie_id=?", (row["movie_id"],)).fetchone()
            if not existing:
                name_year = str(row["movie_id"])
                conn.execute("INSERT OR IGNORE INTO movies(movie_id, name, year, created_at, updated_at) VALUES (?, ?, NULL, ?, ?)", (name_year, name_year, utc_now(), utc_now()))
            conn.execute("INSERT INTO feedback(movie_id, feedback, created_at) VALUES (?, ?, ?)", (row["movie_id"], row.get("feedback", ""), utc_now()))
            count += 1
    return count


def rebuild_database(export_zip: str | Path = "data/letterboxd_export.zip", cache_path: str | Path = "data/tmdb_cache.json", db_path: str | Path = DB_PATH) -> Dict[str, Any]:
    path = Path(db_path)
    if path.exists():
        path.unlink()
    init_db(path)
    result: Dict[str, Any] = {}
    result["letterboxd"] = import_letterboxd_export(export_zip=export_zip, db_path=path)
    result["tmdb_metadata"] = import_tmdb_cache(cache_path=cache_path, db_path=path)
    result["feedback"] = import_feedback_csv(db_path=path)
    return result


def load_data_from_db(db_path: str | Path = DB_PATH) -> Dict[str, pd.DataFrame]:
    init_db(db_path)
    with connect(db_path) as conn:
        movies_sql = "SELECT movie_id, name AS Name, year AS Year, letterboxd_uri AS 'Letterboxd URI' FROM movies"
        ratings = pd.read_sql_query("SELECT m.name AS Name, m.year AS Year, r.rating AS Rating, r.rated_at AS Date, m.letterboxd_uri AS 'Letterboxd URI', m.movie_id FROM ratings r JOIN movies m USING(movie_id)", conn)
        watched = pd.read_sql_query("SELECT DISTINCT m.name AS Name, m.year AS Year, MIN(w.watched_date) AS Date, m.letterboxd_uri AS 'Letterboxd URI', m.movie_id FROM watched_events w JOIN movies m USING(movie_id) GROUP BY m.movie_id", conn)
        diary = pd.read_sql_query("SELECT m.name AS Name, m.year AS Year, w.watched_date AS 'Watched Date', w.rewatch AS Rewatch, w.rating AS Rating, w.review_text AS Review, m.letterboxd_uri AS 'Letterboxd URI', m.movie_id FROM watched_events w JOIN movies m USING(movie_id)", conn)
        watchlist = pd.read_sql_query("SELECT m.name AS Name, m.year AS Year, wl.added_at AS Date, m.letterboxd_uri AS 'Letterboxd URI', m.movie_id FROM watchlist wl JOIN movies m USING(movie_id) WHERE wl.active=1", conn)
        likes = pd.read_sql_query("SELECT m.name AS Name, m.year AS Year, l.liked_at AS Date, m.letterboxd_uri AS 'Letterboxd URI', m.movie_id FROM likes l JOIN movies m USING(movie_id)", conn)
        lists = pd.read_sql_query("SELECT le.position AS Position, m.name AS Name, m.year AS Year, m.letterboxd_uri AS URL, le.list_name AS source_list, m.movie_id FROM list_entries le JOIN movies m USING(movie_id)", conn)
    return {"ratings": ratings, "watched": watched, "diary": diary, "watchlist": watchlist, "likes": likes, "lists": lists}


def load_metadata_from_db(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        df = pd.read_sql_query("SELECT m.name AS name, m.year AS year, md.* FROM movie_metadata md JOIN movies m USING(movie_id)", conn)
    for col in ["genres", "directors", "writers", "cast", "keywords", "countries", "languages", "moods"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else ([] if pd.isna(x) else x))
    return df


def load_feedback_from_db(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query("SELECT movie_id, feedback, created_at FROM feedback", conn)


def database_status(db_path: str | Path = DB_PATH) -> Dict[str, Any]:
    if not Path(db_path).exists():
        return {"exists": False}
    init_db(db_path)
    with connect(db_path) as conn:
        tables = ["movies", "ratings", "watched_events", "watchlist", "likes", "list_entries", "movie_metadata", "feedback", "rating_history", "sync_runs"]
        status = {"exists": True, "path": str(Path(db_path)), "size_mb": round(Path(db_path).stat().st_size / 1024 / 1024, 2)}
        for t in tables:
            status[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        latest = conn.execute("SELECT source, completed_at, status, message FROM sync_runs ORDER BY id DESC LIMIT 5").fetchall()
        status["recent_sync_runs"] = [dict(r) for r in latest]
    return status


def _json(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "[]"
    if isinstance(value, str):
        return value if value.startswith("[") else json.dumps([value])
    try:
        return json.dumps(list(value), ensure_ascii=False)
    except Exception:
        return json.dumps([])


def _upsert_metadata(conn: sqlite3.Connection, mid: str, entry: Dict[str, Any]) -> None:
    now = utc_now()
    poster_path = entry.get("poster_path")
    poster_url = entry.get("poster_url") or (f"https://image.tmdb.org/t/p/w342{poster_path}" if poster_path else "")
    conn.execute(
        """
        INSERT INTO movie_metadata(movie_id, tmdb_id, tmdb_found, tmdb_title, tmdb_release_date, overview, genres, directors, writers, cast, keywords, countries, languages, moods, runtime, tmdb_vote_average, tmdb_vote_count, tmdb_popularity, poster_path, poster_url, tmdb_url, discovered_from, raw_json, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(movie_id) DO UPDATE SET
          tmdb_id=excluded.tmdb_id, tmdb_found=excluded.tmdb_found, tmdb_title=excluded.tmdb_title, tmdb_release_date=excluded.tmdb_release_date,
          overview=excluded.overview, genres=excluded.genres, directors=excluded.directors, writers=excluded.writers, cast=excluded.cast,
          keywords=excluded.keywords, countries=excluded.countries, languages=excluded.languages, moods=excluded.moods, runtime=excluded.runtime,
          tmdb_vote_average=excluded.tmdb_vote_average, tmdb_vote_count=excluded.tmdb_vote_count, tmdb_popularity=excluded.tmdb_popularity,
          poster_path=excluded.poster_path, poster_url=excluded.poster_url, tmdb_url=excluded.tmdb_url, discovered_from=excluded.discovered_from,
          raw_json=excluded.raw_json, updated_at=excluded.updated_at
        """,
        (
            mid, entry.get("tmdb_id"), int(bool(entry.get("tmdb_found"))), entry.get("tmdb_title"), entry.get("tmdb_release_date"), entry.get("overview", ""),
            _json(entry.get("genres")), _json(entry.get("directors")), _json(entry.get("writers")), _json(entry.get("cast")), _json(entry.get("keywords")),
            _json(entry.get("countries")), _json(entry.get("languages")), _json(entry.get("moods")), entry.get("runtime"), entry.get("tmdb_vote_average"),
            entry.get("tmdb_vote_count"), entry.get("tmdb_popularity"), poster_path, poster_url, entry.get("tmdb_url"), entry.get("discovered_from"), json.dumps(entry, ensure_ascii=False), now,
        ),
    )


def _start_run(conn: sqlite3.Connection, source: str) -> int:
    cur = conn.execute("INSERT INTO sync_runs(source, started_at, status) VALUES (?, ?, ?)", (source, utc_now(), "running"))
    return int(cur.lastrowid)


def _finish_run(conn: sqlite3.Connection, run_id: int, status: str, message: str = "") -> None:
    conn.execute("UPDATE sync_runs SET completed_at=?, status=?, message=? WHERE id=?", (utc_now(), status, message, run_id))


def save_curated_week(
    anchor_movie_id: str,
    anchor_name: str,
    style: str,
    curated_df: pd.DataFrame,
    label: str = "",
    db_path: str | Path = DB_PATH,
) -> int:
    init_db(db_path)
    movies_json = curated_df.to_json(orient="records", force_ascii=False)
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO curated_weeks(anchor_movie_id, anchor_name, style, total_movies, label, movies_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (anchor_movie_id, anchor_name, style, len(curated_df), label or "", movies_json, utc_now()),
        )
        return int(cur.lastrowid)


def load_curated_weeks(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, anchor_name, style, total_movies, label, created_at FROM curated_weeks ORDER BY created_at DESC"
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["id", "anchor_name", "style", "total_movies", "label", "created_at"])
    return pd.DataFrame([dict(r) for r in rows])


def load_curated_week(week_id: int, db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT movies_json FROM curated_weeks WHERE id=?", (week_id,)).fetchone()
    if row is None:
        return pd.DataFrame()
    return pd.read_json(row["movies_json"], orient="records")


def save_feedback_to_db(movie_id_value: str, feedback_value: str, db_path: str | Path = DB_PATH) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        existing = conn.execute("SELECT movie_id FROM movies WHERE movie_id=?", (movie_id_value,)).fetchone()
        if not existing:
            conn.execute("INSERT OR IGNORE INTO movies(movie_id, name, year, created_at, updated_at) VALUES (?, ?, NULL, ?, ?)", (movie_id_value, movie_id_value, utc_now(), utc_now()))
        already = conn.execute("SELECT 1 FROM feedback WHERE movie_id=? AND feedback=?", (movie_id_value, feedback_value)).fetchone()
        if not already:
            conn.execute("INSERT INTO feedback(movie_id, feedback, created_at) VALUES (?, ?, ?)", (movie_id_value, feedback_value, utc_now()))


def remove_feedback_from_db(movie_id_value: str, labels: list, db_path: str | Path = DB_PATH) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        for label in labels:
            conn.execute("DELETE FROM feedback WHERE movie_id=? AND feedback=?", (movie_id_value, label))
