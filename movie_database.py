from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

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
                record_status TEXT NOT NULL DEFAULT 'active',
                status_reason TEXT,
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
                date_kind TEXT NOT NULL DEFAULT 'exact',
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
                imdb_id TEXT,
                imdb_rating REAL,
                imdb_votes INTEGER,
                letterboxd_rating REAL,
                content_type TEXT NOT NULL DEFAULT 'film',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie_id TEXT REFERENCES movies(movie_id) ON DELETE CASCADE,
                feedback TEXT NOT NULL,
                scope TEXT,
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

            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie_id TEXT REFERENCES movies(movie_id) ON DELETE CASCADE,
                category TEXT NOT NULL,
                rating REAL NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL
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

            CREATE TABLE IF NOT EXISTS profiles (
                profile_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                is_primary INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS watchlist_context (
                movie_id TEXT NOT NULL REFERENCES movies(movie_id) ON DELETE CASCADE,
                profile_id TEXT NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
                priority INTEGER,
                reason TEXT,
                desired_timeframe TEXT,
                watch_mode TEXT NOT NULL DEFAULT 'solo',
                updated_at TEXT NOT NULL,
                PRIMARY KEY(movie_id, profile_id)
            );

            CREATE TABLE IF NOT EXISTS title_availability (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie_id TEXT NOT NULL REFERENCES movies(movie_id) ON DELETE CASCADE,
                region TEXT NOT NULL DEFAULT 'BE',
                provider TEXT NOT NULL,
                format TEXT NOT NULL DEFAULT 'streaming',
                access_type TEXT NOT NULL DEFAULT 'subscription',
                subtitle_languages TEXT,
                expires_on TEXT,
                source TEXT NOT NULL DEFAULT 'manual',
                active INTEGER NOT NULL DEFAULT 1,
                last_checked TEXT NOT NULL,
                UNIQUE(movie_id, region, provider, format, access_type)
            );

            CREATE TABLE IF NOT EXISTS watch_context (
                watched_event_id INTEGER PRIMARY KEY REFERENCES watched_events(id) ON DELETE CASCADE,
                mood TEXT,
                companions TEXT,
                setting TEXT,
                note TEXT,
                would_rewatch INTEGER,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS movie_notes (
                movie_id TEXT PRIMARY KEY REFERENCES movies(movie_id) ON DELETE CASCADE,
                note TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS title_identity_review (
                movie_id TEXT PRIMARY KEY REFERENCES movies(movie_id) ON DELETE CASCADE,
                review_status TEXT NOT NULL DEFAULT 'pending',
                candidate_tmdb_id INTEGER,
                note TEXT,
                reviewed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_watched_movie ON watched_events(movie_id);
            CREATE INDEX IF NOT EXISTS idx_ratings_rating ON ratings(rating);
            CREATE INDEX IF NOT EXISTS idx_metadata_tmdb_id ON movie_metadata(tmdb_id);
            CREATE INDEX IF NOT EXISTS idx_reflections_movie ON reflections(movie_id);
            CREATE INDEX IF NOT EXISTS idx_availability_movie ON title_availability(movie_id, active);
            """
        )
        # Migration: add feedback.scope to databases created before scope tracking existed.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(feedback)").fetchall()}
        if "scope" not in cols:
            conn.execute("ALTER TABLE feedback ADD COLUMN scope TEXT")

        # Migration: add IMDb/Letterboxd rating columns to databases created before external-rating tracking existed.
        meta_cols = {row[1] for row in conn.execute("PRAGMA table_info(movie_metadata)").fetchall()}
        for col, coltype in [
            ("imdb_id", "TEXT"), ("imdb_rating", "REAL"), ("imdb_votes", "INTEGER"), ("letterboxd_rating", "REAL"),
            ("content_type", "TEXT NOT NULL DEFAULT 'film'"),
        ]:
            if col not in meta_cols:
                conn.execute(f"ALTER TABLE movie_metadata ADD COLUMN {col} {coltype}")

        movie_cols = {row[1] for row in conn.execute("PRAGMA table_info(movies)").fetchall()}
        for col, coltype in [
            ("record_status", "TEXT NOT NULL DEFAULT 'active'"),
            ("status_reason", "TEXT"),
        ]:
            if col not in movie_cols:
                conn.execute(f"ALTER TABLE movies ADD COLUMN {col} {coltype}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_status ON movies(record_status)")

        watched_cols = {row[1] for row in conn.execute("PRAGMA table_info(watched_events)").fetchall()}
        if "date_kind" not in watched_cols:
            conn.execute("ALTER TABLE watched_events ADD COLUMN date_kind TEXT NOT NULL DEFAULT 'exact'")

        # watched.csv records when a title was marked watched on Letterboxd. It is useful
        # evidence that the title was seen, but it is not a diary viewing date.
        conn.execute(
            "UPDATE watched_events SET watched_date=NULL, date_kind='unknown' "
            "WHERE source='letterboxd_export' AND date_kind<>'unknown'"
        )
        conn.execute("UPDATE watched_events SET date_kind='exact' WHERE source IN ('letterboxd_diary','letterboxd_rss')")

        conn.execute(
            "INSERT OR IGNORE INTO profiles(profile_id, name, is_primary, created_at) VALUES ('me', 'Me', 1, ?)",
            (utc_now(),),
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
                    conn.execute("INSERT OR IGNORE INTO watched_events(movie_id, watched_date, rewatch, rating, review_text, source, source_event_id, date_kind, created_at) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?)", (mid, 0, None, "", "letterboxd_export", f"watched:{mid}", "unknown", utc_now()))
                    counts["watched_events"] += 1

            diary = data.get("diary", pd.DataFrame()).copy()
            if not diary.empty:
                for _, row in diary.iterrows():
                    mid = upsert_movie(conn, row["Name"], row["Year"], uri=row.get("Letterboxd URI", ""))
                    rating = pd.to_numeric(row.get("Rating"), errors="coerce")
                    rewatch = str(row.get("Rewatch", "")).lower() in {"yes", "true", "1"}
                    conn.execute("INSERT OR IGNORE INTO watched_events(movie_id, watched_date, rewatch, rating, review_text, source, source_event_id, date_kind, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (mid, row.get("Watched Date", row.get("Date", "")), int(rewatch), None if pd.isna(rating) else float(rating), row.get("Review", ""), "letterboxd_diary", f"diary:{mid}:{row.get('Watched Date', row.get('Date',''))}:{row.get('Rating','')}", "exact", utc_now()))
                    counts["watched_events"] += 1

            watchlist = data.get("watchlist", pd.DataFrame()).copy()
            # A full export is authoritative for current watchlist membership.
            conn.execute("UPDATE watchlist SET active=0,removed_at=?,updated_at=? WHERE active=1", (utc_now(), utc_now()))
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
                    "(movie_id, watched_date, rewatch, rating, review_text, source, source_event_id, date_kind, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        mid, str(row.get("Watched Date", "")), int(rewatch),
                        None if pd.isna(rating) else float(rating),
                        "", "letterboxd_rss", event_id, "exact", utc_now(),
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
            scope = row.get("scope") if "scope" in fb.columns and pd.notna(row.get("scope")) else "recommendation"
            conn.execute("INSERT INTO feedback(movie_id, feedback, scope, created_at) VALUES (?, ?, ?, ?)", (row["movie_id"], row.get("feedback", ""), scope, utc_now()))
            count += 1
    return count


def rebuild_database(export_zip: str | Path = "data/letterboxd_export.zip", cache_path: str | Path = "data/tmdb_cache.json", db_path: str | Path = DB_PATH) -> Dict[str, Any]:
    path = Path(db_path)
    preserved: Dict[str, list[Dict[str, Any]]] = {}
    watch_context_rows: list[Dict[str, Any]] = []
    if path.exists():
        init_db(path)
        with connect(path) as conn:
            for table in [
                "profiles", "watchlist_context", "title_availability", "movie_notes",
                "reflections", "curated_weeks", "title_identity_review",
            ]:
                preserved[table] = [dict(row) for row in conn.execute(f"SELECT * FROM {table}").fetchall()]
            watch_context_rows = [dict(row) for row in conn.execute(
                "SELECT w.movie_id,w.source_event_id,c.mood,c.companions,c.setting,c.note,c.would_rewatch,c.updated_at "
                "FROM watch_context c JOIN watched_events w ON w.id=c.watched_event_id"
            ).fetchall()]
        path.unlink()
    init_db(path)
    result: Dict[str, Any] = {}
    result["letterboxd"] = import_letterboxd_export(export_zip=export_zip, db_path=path)
    result["tmdb_metadata"] = import_tmdb_cache(cache_path=cache_path, db_path=path)
    result["feedback"] = import_feedback_csv(db_path=path)
    restored = 0
    with connect(path) as conn:
        for table, rows in preserved.items():
            valid_cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for row in rows:
                payload = {k: v for k, v in row.items() if k in valid_cols}
                if not payload:
                    continue
                cols = list(payload)
                placeholders = ",".join("?" for _ in cols)
                conn.execute(
                    f"INSERT OR REPLACE INTO {table}({','.join(cols)}) VALUES ({placeholders})",
                    tuple(payload[c] for c in cols),
                )
                restored += 1
        for row in watch_context_rows:
            event = conn.execute(
                "SELECT id FROM watched_events WHERE movie_id=? AND source_event_id=? ORDER BY id DESC LIMIT 1",
                (row["movie_id"], row["source_event_id"]),
            ).fetchone()
            if event:
                conn.execute(
                    "INSERT OR REPLACE INTO watch_context(watched_event_id,mood,companions,setting,note,would_rewatch,updated_at) VALUES (?,?,?,?,?,?,?)",
                    (event["id"], row["mood"], row["companions"], row["setting"], row["note"], row["would_rewatch"], row["updated_at"]),
                )
                restored += 1
    result["personal_records_restored"] = restored
    result["list_artifacts_quarantined"] = quarantine_list_artifacts(db_path=path)
    return result


def load_data_from_db(db_path: str | Path = DB_PATH) -> Dict[str, pd.DataFrame]:
    init_db(db_path)
    with connect(db_path) as conn:
        active = "COALESCE(m.record_status, 'active')='active'"
        ratings = pd.read_sql_query(f"SELECT m.name AS Name, m.year AS Year, r.rating AS Rating, r.rated_at AS Date, m.letterboxd_uri AS 'Letterboxd URI', m.movie_id FROM ratings r JOIN movies m USING(movie_id) WHERE {active}", conn)
        watched = pd.read_sql_query(f"SELECT m.name AS Name, m.year AS Year, MIN(CASE WHEN w.date_kind='exact' THEN w.watched_date END) AS Date, m.letterboxd_uri AS 'Letterboxd URI', m.movie_id FROM watched_events w JOIN movies m USING(movie_id) WHERE {active} GROUP BY m.movie_id", conn)
        diary = pd.read_sql_query(f"SELECT m.name AS Name, m.year AS Year, w.watched_date AS 'Watched Date', w.date_kind AS 'Date Kind', w.rewatch AS Rewatch, w.rating AS Rating, w.review_text AS Review, m.letterboxd_uri AS 'Letterboxd URI', m.movie_id, w.id AS watched_event_id FROM watched_events w JOIN movies m USING(movie_id) WHERE {active} AND w.date_kind='exact'", conn)
        watchlist = pd.read_sql_query(f"SELECT m.name AS Name, m.year AS Year, wl.added_at AS Date, m.letterboxd_uri AS 'Letterboxd URI', m.movie_id,MAX(wc.priority) AS priority,GROUP_CONCAT(DISTINCT wc.watch_mode) AS watch_modes,GROUP_CONCAT(DISTINCT wc.reason) AS intent_reasons FROM watchlist wl JOIN movies m USING(movie_id) LEFT JOIN watchlist_context wc USING(movie_id) WHERE wl.active=1 AND {active} GROUP BY m.movie_id", conn)
        likes = pd.read_sql_query(f"SELECT m.name AS Name, m.year AS Year, l.liked_at AS Date, m.letterboxd_uri AS 'Letterboxd URI', m.movie_id FROM likes l JOIN movies m USING(movie_id) WHERE {active}", conn)
        lists = pd.read_sql_query(f"SELECT le.position AS Position, m.name AS Name, m.year AS Year, m.letterboxd_uri AS URL, le.list_name AS source_list, m.movie_id FROM list_entries le JOIN movies m USING(movie_id) WHERE {active}", conn)
    return {"ratings": ratings, "watched": watched, "diary": diary, "watchlist": watchlist, "likes": likes, "lists": lists}


def load_metadata_from_db(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        df = pd.read_sql_query("SELECT m.name AS name, m.year AS year, md.* FROM movie_metadata md JOIN movies m USING(movie_id) WHERE COALESCE(m.record_status, 'active')='active'", conn)
    for col in ["genres", "directors", "writers", "cast", "keywords", "countries", "languages", "moods"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else ([] if pd.isna(x) else x))
    return df


def load_feedback_from_db(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query("SELECT movie_id, feedback, COALESCE(scope, 'recommendation') AS scope, created_at FROM feedback", conn)


def load_rating_history_from_db(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query("SELECT movie_id, old_rating, new_rating, changed_at, source FROM rating_history", conn)


def load_profiles(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query("SELECT profile_id, name, is_primary, created_at FROM profiles ORDER BY is_primary DESC, name", conn)


def save_profile(profile_id: str, name: str, is_primary: bool = False, db_path: str | Path = DB_PATH) -> None:
    init_db(db_path)
    pid = str(profile_id).strip().lower().replace(" ", "-")
    if not pid or not str(name).strip():
        raise ValueError("Profile id and name are required.")
    with connect(db_path) as conn:
        if is_primary:
            conn.execute("UPDATE profiles SET is_primary=0")
        conn.execute(
            "INSERT INTO profiles(profile_id,name,is_primary,created_at) VALUES (?,?,?,?) "
            "ON CONFLICT(profile_id) DO UPDATE SET name=excluded.name,is_primary=excluded.is_primary",
            (pid, str(name).strip(), int(is_primary), utc_now()),
        )


def save_watchlist_context(
    movie_id_value: str, profile_id: str = "me", priority: int | None = None,
    reason: str = "", desired_timeframe: str = "", watch_mode: str = "solo",
    db_path: str | Path = DB_PATH,
) -> None:
    init_db(db_path)
    if priority is not None and not 1 <= int(priority) <= 5:
        raise ValueError("Priority must be between 1 and 5.")
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO watchlist_context(movie_id,profile_id,priority,reason,desired_timeframe,watch_mode,updated_at) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(movie_id,profile_id) DO UPDATE SET "
            "priority=excluded.priority,reason=excluded.reason,desired_timeframe=excluded.desired_timeframe,"
            "watch_mode=excluded.watch_mode,updated_at=excluded.updated_at",
            (movie_id_value, profile_id, priority, reason.strip(), desired_timeframe.strip(), watch_mode, utc_now()),
        )


def set_watchlist_active(movie_id_value: str, active: bool, db_path: str | Path = DB_PATH) -> None:
    init_db(db_path)
    now = utc_now()
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE watchlist SET active=?,removed_at=?,updated_at=? WHERE movie_id=?",
            (int(active), None if active else now, now, movie_id_value),
        )


def load_watchlist_context(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT wc.*,m.name AS Name,m.year AS Year,p.name AS profile_name "
            "FROM watchlist_context wc JOIN movies m USING(movie_id) JOIN profiles p USING(profile_id)", conn,
        )


def save_availability(
    movie_id_value: str, provider: str, region: str = "BE", format_value: str = "streaming",
    access_type: str = "subscription", subtitle_languages: str = "", expires_on: str = "",
    source: str = "manual", active: bool = True, db_path: str | Path = DB_PATH,
) -> None:
    init_db(db_path)
    if not provider.strip():
        raise ValueError("Provider or shelf name is required.")
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO title_availability(movie_id,region,provider,format,access_type,subtitle_languages,expires_on,source,active,last_checked) "
            "VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(movie_id,region,provider,format,access_type) DO UPDATE SET "
            "subtitle_languages=excluded.subtitle_languages,expires_on=excluded.expires_on,source=excluded.source,"
            "active=excluded.active,last_checked=excluded.last_checked",
            (movie_id_value, region.upper(), provider.strip(), format_value, access_type,
             subtitle_languages.strip(), expires_on or None, source, int(active), utc_now()),
        )


def load_availability(db_path: str | Path = DB_PATH, active_only: bool = True) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        where = "WHERE a.active=1 AND (a.expires_on IS NULL OR a.expires_on='' OR date(a.expires_on)>=date('now'))" if active_only else ""
        return pd.read_sql_query(
            f"SELECT a.*,m.name AS Name,m.year AS Year FROM title_availability a "
            f"JOIN movies m USING(movie_id) {where} ORDER BY m.name,a.provider", conn,
        )


def set_content_type(movie_id_value: str, content_type: str, db_path: str | Path = DB_PATH) -> None:
    allowed = {"film", "short", "documentary", "tv_series", "miniseries", "concert_film"}
    if content_type not in allowed:
        raise ValueError(f"Unsupported content type: {content_type}")
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("UPDATE movie_metadata SET content_type=?,updated_at=? WHERE movie_id=?", (content_type, utc_now(), movie_id_value))


def save_watch_context(
    watched_event_id: int, mood: str = "", companions: str = "", setting: str = "",
    note: str = "", would_rewatch: bool | None = None, db_path: str | Path = DB_PATH,
) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO watch_context(watched_event_id,mood,companions,setting,note,would_rewatch,updated_at) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(watched_event_id) DO UPDATE SET mood=excluded.mood,"
            "companions=excluded.companions,setting=excluded.setting,note=excluded.note,"
            "would_rewatch=excluded.would_rewatch,updated_at=excluded.updated_at",
            (int(watched_event_id), mood.strip(), companions.strip(), setting.strip(), note.strip(),
             None if would_rewatch is None else int(would_rewatch), utc_now()),
        )


def load_watch_context(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT c.*,w.movie_id,w.watched_date,m.name AS Name,m.year AS Year FROM watch_context c "
            "JOIN watched_events w ON w.id=c.watched_event_id JOIN movies m ON m.movie_id=w.movie_id", conn,
        )


def save_movie_note(movie_id_value: str, note: str, db_path: str | Path = DB_PATH) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        if not note.strip():
            conn.execute("DELETE FROM movie_notes WHERE movie_id=?", (movie_id_value,))
            return
        existing = conn.execute("SELECT 1 FROM movies WHERE movie_id=?", (movie_id_value,)).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO movies(movie_id,name,year,created_at,updated_at) VALUES (?,?,NULL,?,?)",
                (movie_id_value, movie_id_value, utc_now(), utc_now()),
            )
        conn.execute(
            "INSERT INTO movie_notes(movie_id,note,updated_at) VALUES (?,?,?) "
            "ON CONFLICT(movie_id) DO UPDATE SET note=excluded.note,updated_at=excluded.updated_at",
            (movie_id_value, note.strip(), utc_now()),
        )


def load_movie_notes(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT n.movie_id,m.name AS Name,m.year AS Year,n.note,n.updated_at "
            "FROM movie_notes n JOIN movies m USING(movie_id) ORDER BY n.updated_at DESC", conn,
        )


def identity_review_queue(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT m.movie_id,m.name AS Name,m.year AS Year,m.record_status,m.status_reason,"
            "md.tmdb_id,COALESCE(r.review_status,'pending') AS review_status,r.candidate_tmdb_id,r.note "
            "FROM movies m LEFT JOIN movie_metadata md USING(movie_id) LEFT JOIN title_identity_review r USING(movie_id) "
            "WHERE (md.tmdb_id IS NULL OR m.record_status<>'active' OR r.review_status='pending') ORDER BY m.record_status DESC,m.name", conn,
        )


def set_identity_status(
    movie_id_value: str, status: str, candidate_tmdb_id: int | None = None,
    note: str = "", db_path: str | Path = DB_PATH,
) -> None:
    allowed = {"active", "pending", "quarantined", "ignored"}
    if status not in allowed:
        raise ValueError(f"Unknown identity status: {status}")
    init_db(db_path)
    record_status = "quarantined" if status in {"quarantined", "ignored"} else "active"
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO title_identity_review(movie_id,review_status,candidate_tmdb_id,note,reviewed_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(movie_id) DO UPDATE SET review_status=excluded.review_status,"
            "candidate_tmdb_id=excluded.candidate_tmdb_id,note=excluded.note,reviewed_at=excluded.reviewed_at",
            (movie_id_value, status, candidate_tmdb_id, note.strip(), utc_now()),
        )
        conn.execute(
            "UPDATE movies SET record_status=?,status_reason=?,updated_at=? WHERE movie_id=?",
            (record_status, note.strip() or status, utc_now(), movie_id_value),
        )


def apply_identity_match(movie_id_value: str, metadata: Dict[str, Any], db_path: str | Path = DB_PATH) -> None:
    """Attach reviewer-confirmed TMDb metadata to the existing canonical local title."""
    if not metadata.get("tmdb_found") or not metadata.get("tmdb_id"):
        raise ValueError("Confirmed TMDb metadata is required.")
    init_db(db_path)
    with connect(db_path) as conn:
        _upsert_metadata(conn, movie_id_value, metadata)
        conn.execute(
            "UPDATE movies SET tmdb_id=?,record_status='active',status_reason=NULL,updated_at=? WHERE movie_id=?",
            (metadata["tmdb_id"], utc_now(), movie_id_value),
        )
        conn.execute(
            "INSERT INTO title_identity_review(movie_id,review_status,candidate_tmdb_id,note,reviewed_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(movie_id) DO UPDATE SET review_status=excluded.review_status,"
            "candidate_tmdb_id=excluded.candidate_tmdb_id,note=excluded.note,reviewed_at=excluded.reviewed_at",
            (movie_id_value, "active", metadata["tmdb_id"], "Confirmed TMDb match", utc_now()),
        )


def quarantine_list_artifacts(export_dir: str | Path = "data/letterboxd", db_path: str | Path = DB_PATH) -> int:
    """Quarantine unresolved cache rows whose names exactly match exported list titles."""
    base = Path(export_dir) / "lists"
    titles: set[str] = set()
    for path in base.glob("*.csv") if base.exists() else []:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            if len(lines) >= 3:
                import csv
                header = next(csv.reader([lines[1]]))
                values = next(csv.reader([lines[2]]))
                row = dict(zip(header, values))
                if row.get("Name"):
                    titles.add(row["Name"].strip().casefold())
        except (OSError, UnicodeError, StopIteration):
            continue
    if not titles:
        return 0
    init_db(db_path)
    changed = 0
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT m.movie_id,m.name FROM movies m LEFT JOIN movie_metadata md USING(movie_id) "
            "WHERE m.year IS NULL AND md.tmdb_id IS NULL AND m.record_status='active'"
        ).fetchall()
        for row in rows:
            if row["name"].strip().casefold() in titles:
                conn.execute(
                    "UPDATE movies SET record_status='quarantined',status_reason='Letterboxd list header imported as title',updated_at=? WHERE movie_id=?",
                    (utc_now(), row["movie_id"]),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO title_identity_review(movie_id,review_status,note,reviewed_at) VALUES (?,?,?,?)",
                    (row["movie_id"], "quarantined", "Letterboxd list header imported as title", utc_now()),
                )
                changed += 1
    return changed


def seed_library_context_from_lists(db_path: str | Path = DB_PATH) -> Dict[str, int]:
    """Turn well-known Letterboxd list conventions into editable structured records."""
    init_db(db_path)
    result = {"availability": 0, "joint_intent": 0}
    with connect(db_path) as conn:
        for row in conn.execute(
            "SELECT DISTINCT movie_id,list_name FROM list_entries WHERE lower(list_name) LIKE '%owned%'"
        ).fetchall():
            fmt = "physical"
            cur = conn.execute(
                "INSERT OR IGNORE INTO title_availability(movie_id,region,provider,format,access_type,source,active,last_checked) "
                "VALUES (?,?,?,?,?,'letterboxd_list',1,?)",
                (row["movie_id"], "BE", "Owned", fmt, "owned", utc_now()),
            )
            result["availability"] += cur.rowcount
        conn.execute(
            "INSERT OR IGNORE INTO profiles(profile_id,name,is_primary,created_at) VALUES ('joint','Joint watch',0,?)",
            (utc_now(),),
        )
        for row in conn.execute(
            "SELECT DISTINCT movie_id FROM list_entries WHERE lower(list_name) LIKE '%joint-watchlist%' "
            "OR lower(list_name) LIKE '%wife-needs-to-watch%'"
        ).fetchall():
            cur = conn.execute(
                "INSERT OR IGNORE INTO watchlist_context(movie_id,profile_id,priority,reason,desired_timeframe,watch_mode,updated_at) "
                "VALUES (?,'joint',3,'Imported from shared Letterboxd list','','joint',?)",
                (row["movie_id"], utc_now()),
            )
            result["joint_intent"] += cur.rowcount
    return result


def load_reflections_from_db(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT r.movie_id, m.name AS Name, m.year AS Year, r.category, r.rating, r.note, r.created_at "
            "FROM reflections r JOIN movies m USING(movie_id)",
            conn,
        )


def database_status(db_path: str | Path = DB_PATH) -> Dict[str, Any]:
    if not Path(db_path).exists():
        return {"exists": False}
    init_db(db_path)
    with connect(db_path) as conn:
        tables = ["movies", "ratings", "watched_events", "watchlist", "likes", "list_entries", "movie_metadata", "feedback", "rating_history", "sync_runs", "reflections", "curated_weeks", "profiles", "watchlist_context", "title_availability", "watch_context", "title_identity_review", "movie_notes"]
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
        INSERT INTO movie_metadata(movie_id, tmdb_id, tmdb_found, tmdb_title, tmdb_release_date, overview, genres, directors, writers, cast, keywords, countries, languages, moods, runtime, tmdb_vote_average, tmdb_vote_count, tmdb_popularity, poster_path, poster_url, tmdb_url, discovered_from, raw_json, imdb_id, content_type, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(movie_id) DO UPDATE SET
          tmdb_id=excluded.tmdb_id, tmdb_found=excluded.tmdb_found, tmdb_title=excluded.tmdb_title, tmdb_release_date=excluded.tmdb_release_date,
          overview=excluded.overview, genres=excluded.genres, directors=excluded.directors, writers=excluded.writers, cast=excluded.cast,
          keywords=excluded.keywords, countries=excluded.countries, languages=excluded.languages, moods=excluded.moods, runtime=excluded.runtime,
          tmdb_vote_average=excluded.tmdb_vote_average, tmdb_vote_count=excluded.tmdb_vote_count, tmdb_popularity=excluded.tmdb_popularity,
          poster_path=excluded.poster_path, poster_url=excluded.poster_url, tmdb_url=excluded.tmdb_url, discovered_from=excluded.discovered_from,
          raw_json=excluded.raw_json, imdb_id=excluded.imdb_id, content_type=excluded.content_type, updated_at=excluded.updated_at
        """,
        (
            mid, entry.get("tmdb_id"), int(bool(entry.get("tmdb_found"))), entry.get("tmdb_title"), entry.get("tmdb_release_date"), entry.get("overview", ""),
            _json(entry.get("genres")), _json(entry.get("directors")), _json(entry.get("writers")), _json(entry.get("cast")), _json(entry.get("keywords")),
            _json(entry.get("countries")), _json(entry.get("languages")), _json(entry.get("moods")), entry.get("runtime"), entry.get("tmdb_vote_average"),
            entry.get("tmdb_vote_count"), entry.get("tmdb_popularity"), poster_path, poster_url, entry.get("tmdb_url"), entry.get("discovered_from"), json.dumps(entry, ensure_ascii=False),
            entry.get("imdb_id"), entry.get("content_type", "film"), now,
        ),
    )


def update_omdb_ratings(ratings: Dict[str, Dict[str, Any]], db_path: str | Path = DB_PATH) -> int:
    """Write IMDb rating/vote-count (keyed by movie_id) into existing movie_metadata rows."""
    init_db(db_path)
    updated = 0
    with connect(db_path) as conn:
        for mid, info in ratings.items():
            cur = conn.execute(
                "UPDATE movie_metadata SET imdb_rating=?, imdb_votes=?, updated_at=? WHERE movie_id=?",
                (info.get("imdb_rating"), info.get("imdb_votes"), utc_now(), mid),
            )
            updated += cur.rowcount
    return updated


def update_letterboxd_ratings(ratings: Dict[str, float], db_path: str | Path = DB_PATH) -> int:
    """Write Letterboxd average rating (keyed by movie_id) into existing movie_metadata rows."""
    init_db(db_path)
    updated = 0
    with connect(db_path) as conn:
        for mid, rating in ratings.items():
            cur = conn.execute(
                "UPDATE movie_metadata SET letterboxd_rating=?, updated_at=? WHERE movie_id=?",
                (rating, utc_now(), mid),
            )
            updated += cur.rowcount
    return updated


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


def save_feedback_to_db(movie_id_value: str, feedback_value: str, scope: str = "recommendation", db_path: str | Path = DB_PATH) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        existing = conn.execute("SELECT movie_id FROM movies WHERE movie_id=?", (movie_id_value,)).fetchone()
        if not existing:
            conn.execute("INSERT OR IGNORE INTO movies(movie_id, name, year, created_at, updated_at) VALUES (?, ?, NULL, ?, ?)", (movie_id_value, movie_id_value, utc_now(), utc_now()))
        already = conn.execute("SELECT 1 FROM feedback WHERE movie_id=? AND feedback=?", (movie_id_value, feedback_value)).fetchone()
        if not already:
            conn.execute("INSERT INTO feedback(movie_id, feedback, scope, created_at) VALUES (?, ?, ?, ?)", (movie_id_value, feedback_value, scope, utc_now()))


def remove_feedback_from_db(movie_id_value: str, labels: list, db_path: str | Path = DB_PATH) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        for label in labels:
            conn.execute("DELETE FROM feedback WHERE movie_id=? AND feedback=?", (movie_id_value, label))


def save_reflection(
    movie_id_value: str,
    name: str,
    year: Any,
    category_ratings: Dict[str, Tuple[float, str]],
    overall_rating: float,
    overall_note: str = "",
    db_path: str | Path = DB_PATH,
) -> None:
    """Append category sub-ratings + notes and an overall rating for a film.

    Append-only (mirrors rating_history) — re-reflecting after a rewatch adds new rows rather
    than overwriting old ones. Also upserts the film's official rating in `ratings`, logging to
    `rating_history` only when the value actually changes.
    """
    init_db(db_path)
    now = utc_now()
    with connect(db_path) as conn:
        mid = upsert_movie(conn, name, year)
        for category, (rating, note) in category_ratings.items():
            conn.execute(
                "INSERT INTO reflections(movie_id, category, rating, note, created_at) VALUES (?, ?, ?, ?, ?)",
                (mid, category, float(rating), note or "", now),
            )
        conn.execute(
            "INSERT INTO reflections(movie_id, category, rating, note, created_at) VALUES (?, ?, ?, ?, ?)",
            (mid, "Overall", float(overall_rating), overall_note or "", now),
        )

        old = conn.execute("SELECT rating FROM ratings WHERE movie_id=?", (mid,)).fetchone()
        if old is not None and old["rating"] != float(overall_rating):
            conn.execute(
                "INSERT INTO rating_history(movie_id, old_rating, new_rating, changed_at, source) VALUES (?, ?, ?, ?, ?)",
                (mid, old["rating"], float(overall_rating), now, "reflection"),
            )
        conn.execute(
            "INSERT INTO ratings(movie_id, rating, rated_at, source, updated_at) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(movie_id) DO UPDATE SET"
            "   rating=excluded.rating, source=excluded.source, updated_at=excluded.updated_at",
            (mid, float(overall_rating), now, "reflection", now),
        )


def load_latest_reflection(movie_id_value: str, db_path: str | Path = DB_PATH) -> Dict[str, Any]:
    """Latest rating+note per category (and overall) previously saved for this film."""
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT r.category, r.rating, r.note, r.created_at
            FROM reflections r
            INNER JOIN (
                SELECT category, MAX(id) AS latest_id
                FROM reflections WHERE movie_id=? GROUP BY category
            ) latest ON latest.category = r.category AND latest.latest_id = r.id
            WHERE r.movie_id=?
            """,
            (movie_id_value, movie_id_value),
        ).fetchall()
    categories: Dict[str, Dict[str, Any]] = {}
    overall: Optional[Dict[str, Any]] = None
    for row in rows:
        entry = {"rating": row["rating"], "note": row["note"], "created_at": row["created_at"]}
        if row["category"] == "Overall":
            overall = entry
        else:
            categories[row["category"]] = entry
    return {"categories": categories, "overall": overall}
