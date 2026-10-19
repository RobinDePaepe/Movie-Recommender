"""Typed, UI-independent adapters around the library, ranker, and retriever."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pandas as pd

from movie_recommender.data import movie_database
from movie_recommender.analysis import analysis as taste_analysis
from movie_recommender.recommendations import recommender
from movie_recommender.retrieval.repository import Evidence, PgVectorRepository, RetrievalFilters


@dataclass(frozen=True)
class LibraryQuery:
    """Structured, read-only library query. Unfiltered rows require an explicit sort."""

    title: str = ""
    year: int | None = None
    rating_min: float | None = None
    rating_max: float | None = None
    watched: bool | None = None
    sort_by: str | None = None
    limit: int = 20

    def __post_init__(self) -> None:
        if self.limit < 1 or self.limit > 100:
            raise ValueError("limit must be between 1 and 100")
        if self.sort_by not in (None, "recent_watched", "oldest_watched"):
            raise ValueError("sort_by must be recent_watched or oldest_watched")
        if self.rating_min is not None and self.rating_max is not None and self.rating_min > self.rating_max:
            raise ValueError("rating_min cannot exceed rating_max")


@dataclass(frozen=True)
class LibraryRecord:
    movie_id: str
    title: str
    year: int | None
    rating: float | None
    watched: bool
    letterboxd_uri: str | None = None
    watched_date: str | None = None
    source_id: str | None = None
    source_type: str = "library_record"


@dataclass(frozen=True)
class SourceReference:
    """Stable citation contract shared by orchestration and future clients."""

    source_type: str
    source_id: str
    movie_id: str
    title: str | None
    year: int | None
    excerpt: str
    source: str | None = None
    content_type: str | None = None
    watched_date: str | None = None
    rating: float | None = None
    uri: str | None = None
    tmdb_id: int | None = None


class LibraryQueryService:
    """Reads exact movie/rating/watch facts through the existing SQLite adapter."""

    def __init__(self, data_loader: Callable[[], Mapping[str, pd.DataFrame]] | None = None):
        self._data_loader = data_loader or movie_database.load_data_from_db

    def search(self, query: LibraryQuery) -> list[LibraryRecord]:
        data = self._data_loader()
        ratings = data.get("ratings", pd.DataFrame()).copy()
        watched = data.get("watched", pd.DataFrame()).copy()
        diary = data.get("diary", pd.DataFrame()).copy()
        if query.sort_by in ("recent_watched", "oldest_watched"):
            return self._watched_events(diary, query)
        if not any((query.title, query.year is not None, query.rating_min is not None,
                    query.rating_max is not None, query.watched is not None)):
            return []
        if "movie_id" not in ratings.columns and "movie_id" not in watched.columns:
            return []
        # Merge on the canonical key, keeping a movie even if it only appears in one source.
        cols = [c for c in ("movie_id", "Name", "Year", "Rating", "Letterboxd URI") if c in ratings.columns]
        rows = ratings[cols].copy() if cols else pd.DataFrame(columns=["movie_id"])
        if not watched.empty and "movie_id" in watched:
            wcols = [c for c in ("movie_id", "Name", "Year", "Letterboxd URI") if c in watched.columns]
            watched_rows = watched[wcols].drop_duplicates("movie_id")
            if rows.empty:
                rows = watched_rows
            else:
                rows = rows.merge(watched_rows, on="movie_id", how="outer", suffixes=("", "_watched"))
                for column in ("Name", "Year", "Letterboxd URI"):
                    alt = f"{column}_watched"
                    if alt in rows:
                        rows[column] = rows.get(column, pd.Series(index=rows.index, dtype=object)).fillna(rows[alt])
                        rows.drop(columns=[alt], inplace=True)
        if rows.empty:
            return []
        watched_ids = set(watched.get("movie_id", pd.Series(dtype=str)).dropna().astype(str))
        if query.title:
            title_col = rows.get("Name", pd.Series("", index=rows.index)).fillna("").astype(str)
            rows = rows[title_col.str.casefold().str.contains(query.title.strip().casefold(), regex=False)]
        if query.year is not None and "Year" in rows:
            rows = rows[pd.to_numeric(rows["Year"], errors="coerce") == query.year]
        if query.rating_min is not None or query.rating_max is not None:
            if "Rating" not in rows:
                return []
            rating = pd.to_numeric(rows["Rating"], errors="coerce")
            if query.rating_min is not None:
                rows = rows[rating >= query.rating_min]
            if query.rating_max is not None:
                rows = rows[rating <= query.rating_max]
        if query.watched is not None:
            rows = rows[rows["movie_id"].astype(str).isin(watched_ids) == query.watched]
        records = []
        for _, row in rows.head(query.limit).iterrows():
            mid = str(row["movie_id"])
            rating = pd.to_numeric(row.get("Rating"), errors="coerce")
            year = pd.to_numeric(row.get("Year"), errors="coerce")
            records.append(LibraryRecord(
                movie_id=mid, title=str(row.get("Name") or mid),
                year=None if pd.isna(year) else int(year),
                rating=None if pd.isna(rating) else float(rating), watched=mid in watched_ids,
                letterboxd_uri=row.get("Letterboxd URI") or None,
            ))
        return records

    @staticmethod
    def _watched_events(diary: pd.DataFrame, query: LibraryQuery) -> list[LibraryRecord]:
        if diary.empty or "movie_id" not in diary or "Watched Date" not in diary:
            return []
        rows = diary.copy()
        rows["_watched_at"] = pd.to_datetime(rows["Watched Date"], errors="coerce")
        rows = rows.dropna(subset=["_watched_at"])
        if query.title:
            titles = rows.get("Name", pd.Series("", index=rows.index)).fillna("").astype(str)
            rows = rows[titles.str.casefold().str.contains(query.title.strip().casefold(), regex=False)]
        if query.year is not None:
            rows = rows[pd.to_numeric(rows.get("Year"), errors="coerce") == query.year]
        if query.rating_min is not None or query.rating_max is not None:
            ratings = pd.to_numeric(rows.get("Rating"), errors="coerce")
            if query.rating_min is not None:
                rows = rows[ratings >= query.rating_min]
            if query.rating_max is not None:
                rows = rows[ratings <= query.rating_max]
        rows = rows.sort_values("_watched_at", ascending=query.sort_by == "oldest_watched")
        rows = rows.drop_duplicates(subset="movie_id", keep="first")
        records = []
        for _, row in rows.head(query.limit).iterrows():
            rating = pd.to_numeric(row.get("Rating"), errors="coerce")
            year = pd.to_numeric(row.get("Year"), errors="coerce")
            event_id = row.get("watched_event_id")
            records.append(LibraryRecord(
                movie_id=str(row["movie_id"]), title=str(row.get("Name") or row["movie_id"]),
                year=None if pd.isna(year) else int(year),
                rating=None if pd.isna(rating) else float(rating), watched=True,
                letterboxd_uri=row.get("Letterboxd URI") or None,
                watched_date=str(row.get("Watched Date"))[:10],
                source_id=f"watched_event:{event_id}" if pd.notna(event_id) else None,
            ))
        return records


class ReviewRetriever:
    """Backend-neutral retrieval adapter returning typed corpus evidence."""

    def __init__(self, repository: Any | None = None):
        self.repository = repository or PgVectorRepository()

    def search(self, query: str, *, filters: RetrievalFilters | None = None,
               mode: str = "hybrid", limit: int = 8) -> list[Evidence]:
        return self.repository.search(query, mode=mode, limit=limit, filters=filters)


@dataclass(frozen=True)
class RecommendationRequest:
    mode: str = "watchlist"
    taste_mode: str = "Balanced"
    anchor_movie_id: str | None = None
    limit: int = 10

    def __post_init__(self) -> None:
        if self.mode not in {"watchlist", "outside_watchlist"}:
            raise ValueError("mode must be 'watchlist' or 'outside_watchlist'")
        if self.limit < 1 or self.limit > 100:
            raise ValueError("limit must be between 1 and 100")


class RecommendationService:
    """Calls the deterministic ranker and returns its rank order unchanged."""

    def __init__(self, *, data_loader: Callable[[], Mapping[str, pd.DataFrame]] | None = None,
                 metadata_loader: Callable[[], pd.DataFrame] | None = None,
                 feedback_loader: Callable[[], pd.DataFrame] | None = None,
                 ranker: Callable[..., tuple[pd.DataFrame, pd.DataFrame]] | None = None):
        self._data_loader = data_loader or movie_database.load_data_from_db
        self._metadata_loader = metadata_loader or movie_database.load_metadata_from_db
        self._feedback_loader = feedback_loader or movie_database.load_feedback_from_db
        self._ranker = ranker or recommender.build_recommendations

    def recommend(self, request: RecommendationRequest) -> pd.DataFrame:
        ranked, _profile = self._ranker(
            dict(self._data_loader()), metadata=self._metadata_loader(), mode=request.mode,
            feedback=self._feedback_loader(), taste_mode=request.taste_mode,
            anchor_movie_id=request.anchor_movie_id,
        )
        return ranked.head(request.limit).copy()


class TasteAnalysisService:
    """Thin read-only adapter for the existing deterministic taste analyses."""

    def __init__(self, *, data_loader: Callable[[], Mapping[str, pd.DataFrame]] | None = None,
                 metadata_loader: Callable[[], pd.DataFrame] | None = None,
                 feedback_loader: Callable[[], pd.DataFrame] | None = None):
        self._data_loader = data_loader or movie_database.load_data_from_db
        self._metadata_loader = metadata_loader or movie_database.load_metadata_from_db
        self._feedback_loader = feedback_loader or movie_database.load_feedback_from_db

    def ratings_over_time(self) -> pd.DataFrame:
        data = self._data_loader()
        return taste_analysis.ratings_over_time(data.get("ratings", pd.DataFrame()), data.get("diary", pd.DataFrame()))

    def entity_affinity(self, *, min_count: int = 3) -> pd.DataFrame:
        data = self._data_loader()
        return taste_analysis.entity_affinity_table(data.get("ratings", pd.DataFrame()), self._metadata_loader(), min_count=min_count)

    def feedback_summary(self) -> pd.DataFrame:
        return taste_analysis.feedback_summary(self._feedback_loader())

    def list_affinity(self, *, min_rated: int = 3) -> pd.DataFrame:
        data = self._data_loader()
        return taste_analysis.list_affinity(data.get("lists", pd.DataFrame()), data.get("ratings", pd.DataFrame()), min_rated=min_rated)


def reference_from_evidence(item: Evidence) -> SourceReference:
    return SourceReference(
        source_type="personal_opinion" if item.source.startswith("letterboxd") else "external_metadata",
        source_id=item.chunk_id, movie_id=item.movie_id,
        title=item.movie_name, year=item.movie_year, excerpt=item.text,
        source=item.source, content_type=item.content_type, watched_date=item.watched_date,
        rating=item.rating, uri=item.letterboxd_uri,
        tmdb_id=item.tmdb_id,
    )
