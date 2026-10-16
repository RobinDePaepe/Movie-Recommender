# RAG Architecture

Last reviewed: 2026-10-16.

The primary design goal is to implement the modules in [`RAG architecture.png`](RAG%20architecture.png). The application should answer natural-language questions using the user's Letterboxd history and movie metadata, retrieve the evidence behind those answers, and call the existing deterministic recommender when ranking is needed. Route structured library facts to SQLite and semantic questions about themes or review opinions to text retrieval; combine both paths when a question needs both.

## Target flow

```text
Letterboxd export ─┐
                   ├─> Ingestion and chunking ─┬─> pgvector review store ─┐
TMDb API ──────────┘                            └─> SQLite library ────────┤
                                                                         v
                                                            Agent orchestrator
                                                               /          \
                                                  Chat interface       Evaluation harness
```

The image labels SQLite scoring DB and pgvector store as existing/new components respectively. In the current code, SQLite stores the library, ratings, diary events, feedback, reflections, and related records. Recommendation scores are computed by Python during a request; there is not yet a persistent score database. Treat that box as the existing library and recommender-data store, with score persistence only if a concrete use case requires it.

## Components and current state

| Diagram component | Current state | Target responsibility |
|---|---|---|
| Letterboxd export | Implemented | Source of complete ratings, diary entries, reviews, watchlist, and lists. Imports must be repeatable and preserve source identifiers. |
| TMDb API | Implemented | Enrich titles with plot, genres, credits, keywords, and stable TMDb identity. |
| Ingestion and chunking | Partial foundation | Existing importers normalize export and TMDb data; SQLite diary records can retain review text. Review corpus normalization, chunking, embedding, and refresh-on-change are not implemented as one pipeline. |
| pgvector store | Not implemented | Store review-text embeddings and chunk metadata for semantic retrieval. Keep movie/source IDs that can be joined to SQLite records. Pair vector retrieval with PostgreSQL full-text search for exact titles, names, and phrases. |
| SQLite library/recommender data | Implemented, with boundary to clarify | Keep canonical movie, rating, diary, feedback, reflection, availability, profile, and watchlist data. Scoring is currently calculated in the recommender module rather than stored as durable score rows. |
| Agent orchestrator | Not implemented | Classify a user query, route to structured queries, semantic retrieval, or both, select from narrow application tools, assemble grounded results, and request a response from a configured LLM provider. It should not replace deterministic ranking. Make provider/data-sharing behavior explicit and preserve whether each claim came from personal review evidence, TMDb metadata, or computed results. |
| Chat interface | Not implemented | Accept natural-language questions and show source cards with film title/year, review date or metadata source, short excerpt, and a link to the source record. Handle weak retrieval and service outages clearly; provide explicit follow-up actions such as watchlist add, recommendation anchor, or curated-week start. |
| Evaluation harness | Partial | The app has rating holdout metrics (including MAE, Precision@10, and NDCG). A fixed RAG query set and separate retrieval relevance, citation coverage, answer-grounding, privacy-routing, lifecycle, injection-resistance, and degraded-mode checks are still needed. |

## Module boundaries

- **Import adapters** read Letterboxd exports, Letterboxd RSS overlays, and TMDb responses. They produce normalized records and do not call an LLM.
- **Corpus builder** selects text fields, creates review/metadata chunks, records provenance, and detects additions, edits, and deletions.
- **Data lifecycle** propagates edits and deletions through normalized corpus rows, chunks, embeddings, and retrieval caches. Provide a safe full-index clear/rebuild operation and confirm deleted text cannot still be retrieved.
- **Embedding service** batches and versions embeddings. A model change must be visible so the corpus can be rebuilt consistently.
- **Vector repository** writes, deletes, and searches chunks in pgvector. It returns chunk text, similarity, and provenance rather than an untraceable text blob. Pair semantic retrieval with PostgreSQL full-text search, then fuse or rerank results; pgvector documents hybrid search with Reciprocal Rank Fusion and cross-encoders ([pgvector hybrid search](https://github.com/pgvector/pgvector#hybrid-search)).
- **SQLite repositories and application services** remain the source for structured filters, ratings, watch state, feedback, profiles, and deterministic recommendations.
- **Orchestrator** routes exact facts and filters to SQLite/application services, semantic questions to retrieval, and mixed questions to both. It constructs a bounded evidence context and produces a response with source references. Expose narrow tools and default to read-only behavior.
- **Provider boundary** makes the selected model/provider and whether personal review text is sent off-device visible to the user. Define chat-history retention and a clear-history operation; do not send personal review text to a hosted model unless the user-configured provider/workflow explicitly does so. Support local providers where practical.
- **Degraded behavior** keeps structured search and deterministic recommendations available when the vector store or LLM provider is unavailable. Label which path produced each result and do not imply that a generated answer was grounded when retrieval failed.
- **UI** is a client of application services. It should not contain retrieval logic or direct SQL for the new RAG path.
- **Evaluation** runs offline against fixed queries and held-out data, recording retrieval relevance, citation coverage, answer grounding, ranking metrics, and latency. Evaluate retrieval and generation separately; RAG evaluation research treats them as distinct components ([RAGAS paper](https://arxiv.org/abs/2309.15217)). Use the fixed set as a release gate when changing chunking, models, retrieval, prompts, or ranking integration.
- **Answer feedback** (after the first usable chat flow) records whether an answer was useful and whether its cited evidence was relevant, linked to the query and source IDs for later evaluation.

## Data and retrieval rules

1. Preserve the canonical `movie_id` format and add stable source/event/chunk identifiers. A chunk should be traceable to its source movie and review or metadata field.
2. Keep personal ratings, diary dates, and review text private. Do not send them to a hosted model unless the selected provider and the user-configured workflow explicitly send that request; support local providers where practical.
3. Retrieve review text and plot/genre metadata as separate typed evidence. A user's opinion in a review must not be presented as an objective plot fact.
4. Use structured SQLite filters for facts such as rating, watched state, year, and availability. Use vector retrieval for semantic evidence; combine the two when a query needs both.
5. For recommendations, call the existing ranker with explicit filters and weights, then attach retrieved evidence to explain results. Do not let generated text silently alter scores.
6. Show enough provenance for the user to verify a response: film title/year, review date or source, and a short excerpt where appropriate.
7. Re-importing unchanged data must not duplicate chunks. Changed source text must invalidate and rebuild only its affected chunks and vectors.
8. Treat retrieved reviews and metadata as untrusted content. They provide evidence only and cannot change system instructions or authorize tool use. Keep tools read-only unless a later, explicit user action requires a write; test prompt-injection attempts in imported and retrieved text. See [OWASP LLM application risks](https://owasp.org/www-project-top-10-for-large-language-model-applications/).
9. Keep source claims typed and distinguish personal opinions, external metadata facts, and computed ranking results in both answer wording and the evidence display.
10. After retrieval or generation failures, offer structured search or deterministic ranking when the query permits it; clearly label the fallback and avoid presenting unsupported claims.
11. Any state-changing action suggested by chat (watchlist, anchor, or curated week) requires an explicit user action and confirmation in the interface; ordinary question answering remains read-only.
12. Provide a data-health view for import/sync freshness, identity resolution, review/metadata coverage, and retrieval-index freshness so users can diagnose weak or stale results.

## Delivery sequence

1. Specify and validate the review corpus schema and idempotent import behavior.
2. Implement chunking and embedding refresh, first with a local development corpus.
3. Add pgvector setup, migrations/schema management, and a vector repository with provenance-aware search.
4. Expose retrieval, recommendation, and analysis through application-service interfaces.
5. Implement the orchestrator and define tool contracts, evidence limits, and failure/empty-result behavior.
6. Add the chat interface and evidence display.
7. Add a fixed-query evaluation set with expected evidence; measure retrieval relevance, citation coverage, answer grounding, and ranking regressions separately. Run it as a release gate and use results to tune chunking and retrieval.
8. Add answer usefulness and evidence-relevance feedback, linked to the query and retrieved source IDs, after the first chat flow is usable.
9. Add explicit user-triggered actions from evidence-backed answers, then add data-health reporting for source and index freshness.

Track implementation tasks in [`open-work.md`](open-work.md). Update the component table as each module is completed; keep the diagram and this document aligned when the target changes.
