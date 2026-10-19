from __future__ import annotations

from movie_recommender.retrieval.repository import Evidence, PgVectorRepository, RetrievalFilters


def _evidence(chunk_id: str, *, semantic=None, lexical=None, text=None) -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        record_id=f"record-{chunk_id}",
        movie_id="velvet nebula (2024)",
        movie_name="Velvet Nebula",
        movie_year=2024,
        source="letterboxd_diary",
        source_event_id="event-1",
        content_type="review_text",
        text=text or f"review for {chunk_id}",
        watched_date="2025-04-03",
        rating=4.5,
        rewatch=False,
        tmdb_id=None,
        letterboxd_uri="https://example.invalid/velvet-nebula",
        chunk_index=0,
        chunker_version="sentence-boundary-v1",
        model_id="test/model",
        model_revision="v1",
        score=semantic if semantic is not None else lexical or 0.0,
        semantic_similarity=semantic,
        lexical_score=lexical,
        search_mode="semantic" if semantic is not None else "lexical",
    )


def test_search_returns_empty_for_blank_query_without_connecting():
    assert PgVectorRepository().search("  ") == []


def test_search_rejects_invalid_mode_and_nonpositive_limit():
    repo = PgVectorRepository()
    try:
        repo.search("query", mode="unknown")
    except ValueError as exc:
        assert "mode" in str(exc)
    else:
        raise AssertionError("invalid search mode should fail")
    try:
        repo.search("query", limit=0)
    except ValueError as exc:
        assert "limit" in str(exc)
    else:
        raise AssertionError("nonpositive limit should fail")


def test_where_builds_parameterized_provenance_and_context_filters():
    where, params = PgVectorRepository._where(RetrievalFilters(
        sources=("letterboxd_diary", "tmdb"),
        content_types=("review_text",),
        movie_ids=("velvet nebula (2024)",),
        watched_date_from="2024-01-01",
        watched_date_to="2025-12-31",
        rating_min=3.5,
        rating_max=5.0,
    ))

    assert "c.source = ANY(%s)" in where
    assert "c.content_type = ANY(%s)" in where
    assert "c.movie_id = ANY(%s)" in where
    assert "c.watched_date >= %s" in where
    assert "c.watched_date <= %s" in where
    assert "c.rating >= %s" in where
    assert "c.rating <= %s" in where
    assert params == [
        ["letterboxd_diary", "tmdb"], ["review_text"], ["velvet nebula (2024)"],
        "2024-01-01", "2025-12-31", 3.5, 5.0,
    ]


def test_hybrid_fusion_uses_reciprocal_rank_and_keeps_provenance():
    repo = PgVectorRepository(rrf_constant=60)
    semantic = [
        _evidence("a", semantic=0.92, text="semantic first"),
        _evidence("b", semantic=0.81),
    ]
    lexical = [
        _evidence("b", lexical=0.8, text="lexical first"),
        _evidence("c", lexical=0.2),
    ]

    fused = repo._fuse(semantic, lexical, limit=3)

    assert [row.chunk_id for row in fused] == ["b", "a", "c"]
    assert fused[0].score == (1 / 62) + (1 / 61)
    assert fused[0].semantic_similarity == 0.81
    assert fused[0].lexical_score == 0.8
    assert fused[0].search_mode == "hybrid"
    assert fused[0].movie_id == "velvet nebula (2024)"
    assert fused[0].source_event_id == "event-1"
    assert fused[0].text == "lexical first"


def test_hybrid_search_runs_both_modes_and_expands_candidate_pool(monkeypatch):
    repo = PgVectorRepository()
    calls = []

    def semantic(query, limit, filters):
        calls.append(("semantic", query, limit, filters))
        return [_evidence("a", semantic=0.9)]

    def lexical(query, limit, filters):
        calls.append(("lexical", query, limit, filters))
        return [_evidence("a", lexical=0.7)]

    monkeypatch.setattr(PgVectorRepository, "_semantic", lambda self, *args: semantic(*args))
    monkeypatch.setattr(PgVectorRepository, "_lexical", lambda self, *args: lexical(*args))
    filters = RetrievalFilters(sources=("tmdb",))

    rows = repo.search("  memory  ", mode="hybrid", limit=2, filters=filters)

    assert len(rows) == 1
    assert rows[0].search_mode == "hybrid"
    assert [(call[0], call[1], call[2], call[3]) for call in calls] == [
        ("semantic", "memory", 6, filters), ("lexical", "memory", 6, filters),
    ]
