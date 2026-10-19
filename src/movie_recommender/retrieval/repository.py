"""Provenance-aware semantic, lexical, and hybrid retrieval from pgvector."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Sequence

import numpy as np

from movie_recommender.retrieval.indexing import (
    MODEL_ID,
    MODEL_REVISION,
    LocalSentenceTransformerEmbedder,
    TextEmbedder,
)
from movie_recommender.retrieval.postgres import postgres_dsn

SearchMode = Literal["semantic", "lexical", "hybrid"]


@dataclass(frozen=True)
class RetrievalFilters:
    """Optional structured restrictions applied inside PostgreSQL retrieval."""

    sources: tuple[str, ...] = ()
    content_types: tuple[str, ...] = ()
    movie_ids: tuple[str, ...] = ()
    watched_date_from: str | None = None
    watched_date_to: str | None = None
    rating_min: float | None = None
    rating_max: float | None = None


@dataclass(frozen=True)
class Evidence:
    """A retrieved chunk with stable source/movie provenance and ranking signals."""

    chunk_id: str
    record_id: str
    movie_id: str
    movie_name: str | None
    movie_year: int | None
    source: str
    source_event_id: str
    content_type: str
    text: str
    watched_date: str | None
    rating: float | None
    rewatch: bool | None
    tmdb_id: int | None
    letterboxd_uri: str | None
    chunk_index: int
    chunker_version: str
    model_id: str
    model_revision: str
    score: float
    semantic_similarity: float | None = None
    lexical_score: float | None = None
    search_mode: str = "semantic"


@dataclass(frozen=True)
class PgVectorRepository:
    """Queries the pgvector index; database-specific SQL stays in this module."""

    dsn: str | None = None
    embedder: TextEmbedder | None = field(default=None, compare=False, repr=False)
    rrf_constant: int = 60

    def search(
        self,
        query: str,
        *,
        mode: SearchMode = "hybrid",
        limit: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[Evidence]:
        """Search for evidence, using RRF to combine semantic and lexical rankings."""
        normalized_query = (query or "").strip()
        if not normalized_query:
            return []
        if mode not in ("semantic", "lexical", "hybrid"):
            raise ValueError("mode must be 'semantic', 'lexical', or 'hybrid'")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        active_filters = filters or RetrievalFilters()
        if mode == "lexical":
            return self._lexical(normalized_query, limit, active_filters)
        semantic = self._semantic(normalized_query, limit if mode == "semantic" else max(limit * 3, limit), active_filters)
        if mode == "semantic":
            return semantic[:limit]
        lexical = self._lexical(normalized_query, max(limit * 3, limit), active_filters)
        return self._fuse(semantic, lexical, limit)

    def _connect(self):
        try:
            import psycopg
            from pgvector.psycopg import register_vector
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("Retrieval requires psycopg and pgvector. Install the project dependencies first.") from exc
        conn = psycopg.connect(self.dsn or postgres_dsn(), row_factory=dict_row)
        register_vector(conn)
        return conn

    @staticmethod
    def _where(filters: RetrievalFilters, alias: str = "c") -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, values in (
            ("source", filters.sources),
            ("content_type", filters.content_types),
            ("movie_id", filters.movie_ids),
        ):
            if values:
                clauses.append(f"{alias}.{column} = ANY(%s)")
                params.append(list(values))
        if filters.watched_date_from is not None:
            clauses.append(f"{alias}.watched_date >= %s")
            params.append(filters.watched_date_from)
        if filters.watched_date_to is not None:
            clauses.append(f"{alias}.watched_date <= %s")
            params.append(filters.watched_date_to)
        if filters.rating_min is not None:
            clauses.append(f"{alias}.rating >= %s")
            params.append(filters.rating_min)
        if filters.rating_max is not None:
            clauses.append(f"{alias}.rating <= %s")
            params.append(filters.rating_max)
        return (" AND ".join(clauses) if clauses else "TRUE"), params

    def _semantic(self, query: str, limit: int, filters: RetrievalFilters) -> list[Evidence]:
        backend = self.embedder or LocalSentenceTransformerEmbedder()
        vector = np.asarray(backend.embed([query]), dtype=np.float32)
        if vector.ndim != 2 or vector.shape != (1, 384) or not np.isfinite(vector).all():
            raise ValueError("Query embedder must return one finite 384-dimensional vector.")
        norm = float(np.linalg.norm(vector[0]))
        if norm == 0:
            raise ValueError("Query embedder returned a zero-length vector.")
        vector = vector[0] / norm
        where, filter_params = self._where(filters)
        sql = f"""SELECT c.*, 1 - (c.embedding <=> %s) AS semantic_similarity
                  FROM retrieval_chunks c
                  WHERE ({where}) AND c.model_id=%s AND c.model_revision=%s
                  ORDER BY c.embedding <=> %s
                  LIMIT %s"""
        params = [vector, *filter_params, backend.model_id, backend.model_revision, vector, limit]
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._evidence(row, score=float(row["semantic_similarity"]),
                               semantic=float(row["semantic_similarity"]), mode="semantic")
                for row in rows]

    def _lexical(self, query: str, limit: int, filters: RetrievalFilters) -> list[Evidence]:
        where, filter_params = self._where(filters)
        sql = f"""SELECT c.*, ts_rank_cd(c.search_vector, plainto_tsquery('simple', %s)) AS lexical_score
                  FROM retrieval_chunks c
                  WHERE ({where}) AND c.search_vector @@ plainto_tsquery('simple', %s)
                  ORDER BY lexical_score DESC, c.chunk_id
                  LIMIT %s"""
        params = [query, *filter_params, query, limit]
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._evidence(row, score=float(row["lexical_score"]),
                               lexical=float(row["lexical_score"]), mode="lexical")
                for row in rows]

    @staticmethod
    def _evidence(
        row: Any,
        *,
        score: float,
        mode: str,
        semantic: float | None = None,
        lexical: float | None = None,
    ) -> Evidence:
        return Evidence(
            chunk_id=row["chunk_id"], record_id=row["record_id"], movie_id=row["movie_id"],
            movie_name=row["movie_name"], movie_year=row["movie_year"], source=row["source"],
            source_event_id=row["source_event_id"], content_type=row["content_type"], text=row["text"],
            watched_date=row["watched_date"], rating=row["rating"], rewatch=row["rewatch"],
            tmdb_id=row["tmdb_id"], letterboxd_uri=row["letterboxd_uri"],
            chunk_index=row["chunk_index"], chunker_version=row["chunker_version"],
            model_id=row["model_id"], model_revision=row["model_revision"], score=score,
            semantic_similarity=semantic, lexical_score=lexical, search_mode=mode,
        )

    def _fuse(self, semantic: Sequence[Evidence], lexical: Sequence[Evidence], limit: int) -> list[Evidence]:
        semantic_ranks = {item.chunk_id: rank for rank, item in enumerate(semantic, start=1)}
        lexical_ranks = {item.chunk_id: rank for rank, item in enumerate(lexical, start=1)}
        semantic_by_id = {item.chunk_id: item for item in semantic}
        lexical_by_id = {item.chunk_id: item for item in lexical}
        by_id = {**semantic_by_id, **lexical_by_id}
        fused: list[Evidence] = []
        for chunk_id, item in by_id.items():
            semantic_rank = semantic_ranks.get(chunk_id)
            lexical_rank = lexical_ranks.get(chunk_id)
            score = sum(
                1.0 / (self.rrf_constant + rank)
                for rank in (semantic_rank, lexical_rank) if rank is not None
            )
            fused.append(replace(
                item,
                score=score,
                search_mode="hybrid",
                semantic_similarity=(semantic_by_id[chunk_id].semantic_similarity
                                     if chunk_id in semantic_by_id else None),
                lexical_score=(lexical_by_id[chunk_id].lexical_score
                               if chunk_id in lexical_by_id else None),
            ))
        fused.sort(key=lambda item: (-item.score, item.chunk_id))
        return fused[:limit]


def search(
    query: str,
    *,
    mode: SearchMode = "hybrid",
    limit: int = 10,
    filters: RetrievalFilters | None = None,
    repository: PgVectorRepository | None = None,
) -> list[Evidence]:
    """Convenience function for application services and lightweight callers."""
    return (repository or PgVectorRepository()).search(query, mode=mode, limit=limit, filters=filters)
