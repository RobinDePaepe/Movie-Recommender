# Guided pgvector Store Development

## Status: complete

All five milestones are implemented. The publisher and search tests pass, including the optional lifecycle integration test against a temporary PostgreSQL schema. Operational rebuild, freshness, and recovery steps are documented in the README. This plan is retained as the implementation record; the next work is the application-service/orchestration path in [open-work.md](open-work.md).

## Milestones

1. **Complete — local PostgreSQL setup.** Docker Compose, connection settings, and startup/shutdown/reset instructions are in place.
2. **Complete — schema and migration.** A versioned migration defines the pgvector index, provenance, model version, and full-text search.
3. **Complete — publish the index.** `scripts/publish_retrieval_index.py` syncs active SQLite chunks and embeddings idempotently, preserving IDs and propagating edits/deletions.
4. **Complete — implement retrieval.** `PgVectorRepository` supports semantic, lexical, and hybrid search with provenance-aware evidence and bounded filters.
5. **Complete — verify and document operations.** Focused tests pass; the optional integration test validates repeat publishing, search provenance, edits, and deletions in an isolated PostgreSQL schema. Rebuild, freshness, and recovery steps are documented in the README.

## Test Plan

At each milestone, I’ll suggest the relevant checks and explain what their results mean. The main acceptance checks are repeatable publishing without duplicates, updates and deletions appearing in search correctly, evidence retaining stable source and movie IDs, and the SQLite library remaining usable if PostgreSQL is down.

## Assumptions

- Use Docker Compose for local PostgreSQL and pgvector.
- Keep SQLite as the canonical library and current chunk/embedding staging source.
- Implement only the pgvector store and retrieval foundation in this effort; orchestration and chat are later milestones.
- Provide code in chat for you to apply; do not write to project scripts or files.
