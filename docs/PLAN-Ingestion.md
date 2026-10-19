# Plan: Review ingestion and normalization

## Status: implemented

The SQLite retrieval corpus implementation is complete. Letterboxd review/diary events and TMDb overview/genre records are normalized with stable IDs and idempotent upserts. Chunking, versioned local embeddings, and pgvector publishing/search are tracked in [the RAG architecture](rag-architecture.md) and [the active open-work list](open-work.md).

## Summary

Create a documented, SQLite-backed retrieval corpus for Letterboxd reviews and diary events, plus TMDb plot and genre metadata. Keep the existing `movie_id` format, preserve source context, and make repeated imports update the same records instead of creating duplicates.

## Implementation changes

- Define a normalized corpus record contract with a record ID, `movie_id`, source and source-event ID, content type, text, relevant diary context, and import/update timestamps. Keep personal review text distinct from TMDb metadata.
- Update full Letterboxd export ingestion to preserve review text and its diary event context. Use a stable Letterboxd identifier when available; otherwise derive an event ID from the movie, watched date, and a deterministic occurrence index. Keep mutable review text and rating out of that fallback identity.
- Include RSS diary events in the same contract. Preserve their event IDs, date, rating, rewatch status, and movie identity; RSS does not provide review text.
- Normalize TMDb overview and genres into typed corpus records linked by canonical `movie_id` and TMDb ID. Re-imports update those records when metadata changes.
- Add unique keys and upsert behavior so reruns and source updates do not duplicate corpus records. Document the fields, source meanings, ID rules, and import behavior in the RAG documentation.

## Test plan

- Import an export containing a review and verify the text, diary date, rating, rewatch status, movie ID, and source ID are preserved.
- Re-import unchanged export and RSS data; verify corpus row counts and IDs remain stable.
- Change review text, rating, or TMDb overview/genres; verify the existing record updates without duplication.
- Cover multiple diary events and reviews for the same film, including same-day events, to verify deterministic IDs and correct context association.
- Import data with missing optional fields and confirm records remain linked and typed without inventing dates or review text.

## Assumptions

- SQLite remains the canonical library and retrieval staging source. Chunking and embeddings are implemented in SQLite; the active index is published to PostgreSQL/pgvector for search.
- The first slice covers full exports, current RSS sync overlays, and TMDb cache imports.
- RSS contributes diary context only; review text comes from the full Letterboxd export.
