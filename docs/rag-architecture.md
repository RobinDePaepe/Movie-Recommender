# RAG Architecture

Last reviewed: 2026-10-19.

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
| Ingestion and chunking | Implemented through SQLite staging embeddings | Letterboxd export/RSS and TMDb cache imports populate a provenance-aware corpus. Nonempty review and TMDb text is chunked and embedded locally with edit/model refresh; the active chunks are published to pgvector for serving. |
| pgvector store | Implemented and locally checked | A versioned migration and idempotent publisher mirror active SQLite chunks into pgvector. `PgVectorRepository` supports semantic, lexical, and reciprocal-rank-fused hybrid search with provenance-aware evidence and structured filters. Focused publisher/search tests and optional isolated PostgreSQL lifecycle checks are documented. |
| SQLite library/recommender data | Implemented, with boundary to clarify | Keep canonical movie, rating, diary, feedback, reflection, availability, profile, and watchlist data. Scoring is currently calculated in the recommender module rather than stored as durable score rows. |
| Agent orchestrator | Implemented (initial service layer; hardening remains) | `movie_recommender.application` exposes typed library, retrieval, recommendation, and taste-analysis services plus a read-only query router. It routes structured, semantic, mixed, recommendation, and analysis requests and returns typed references and evidence-path labels. The classifier and free-text parser are heuristic; some recognized intents are not yet represented by a matching structured query (for example, counts). Generated citation IDs are not yet validated against returned evidence. Provider use is optional; hosted personal-data sharing requires explicit configuration. |
| Chat interface | Implemented (initial slice; reliability work remains) | The Streamlit Chat page calls application services, shows evidence-path/provider disclosures and source cards with title/year, date/source, excerpt, and source link, and handles empty/unavailable paths. Exact library answers remain deterministic, including diary-date sorting for recent-watch queries. Provider choice is explicit, hosted personal evidence requires consent, mismatch flags and conversation history are session-only, and query-answer tools are read-only. |
| Evaluation harness | Partial; RAG evaluation not implemented | Existing rating holdout metrics include MAE, Precision@10, and NDCG. A fixed RAG query set and separate retrieval relevance, citation coverage, answer-grounding, privacy-routing, lifecycle, injection-resistance, and degraded-mode checks are still needed before the chat path can be treated as a release-gated feature. |

## Module boundaries

- **Import adapters** read Letterboxd exports, Letterboxd RSS overlays, and TMDb responses. They produce normalized records and do not call an LLM.
- **Corpus builder and publisher** select text fields, create review/metadata chunks, record provenance, and detect additions, edits, and deletions. `python scripts/build_retrieval_index.py` refreshes SQLite staging chunks and embeddings; `python scripts/publish_retrieval_index.py` applies pending PostgreSQL migrations and mirrors the active index to pgvector.
- **Data lifecycle** propagates edits and deletions through normalized corpus rows, chunks, embeddings, and pgvector publishing. A safe full-index clear/rebuild operation and an evaluated check that deleted text cannot still be retrieved remain outstanding.
- **Embedding service** batches and versions embeddings. The local `all-MiniLM-L6-v2` snapshot is pinned; a model/revision change rebuilds the active vectors consistently. The service boundary can accept a future hosted adapter, but this stage does not send review text to a provider.
- **Vector repository** The publisher handles writes; `PgVectorRepository` searches chunks in pgvector using semantic, lexical, or reciprocal-rank-fused hybrid ranking. It returns chunk text, scores, and provenance rather than an untraceable text blob. pgvector documents hybrid search with Reciprocal Rank Fusion and cross-encoders ([pgvector hybrid search](https://github.com/pgvector/pgvector#hybrid-search)).
- **SQLite repositories and application services** remain the source for structured filters, ratings, watch state, feedback, profiles, and deterministic recommendations.
- **Orchestrator** routes exact facts and filters to SQLite/application services, semantic questions to retrieval, and mixed questions to both. It constructs a bounded evidence context and produces a response with source references. Expose narrow tools and default to read-only behavior.
- **Provider boundary** makes the selected model/provider and whether personal review text is sent off-device visible to the user. Define chat-history retention and a clear-history operation; do not send personal review text to a hosted model unless the user-configured provider/workflow explicitly does so. Support local providers where practical.
- **Degraded behavior** keeps structured search and deterministic recommendations available when the vector store or LLM provider is unavailable. Label which path produced each result and do not imply that a generated answer was grounded when retrieval failed.
- **UI** is a client of application services. It should not contain retrieval logic or direct SQL for the new RAG path.
- **Evaluation (outstanding)** should run offline against fixed queries and held-out data, recording retrieval relevance, citation coverage, answer grounding, ranking metrics, privacy routing, degraded behavior, and latency. Evaluate retrieval and generation separately; RAG evaluation research treats them as distinct components ([RAGAS paper](https://arxiv.org/abs/2309.15217)). Use the fixed set as a release gate when changing chunking, models, retrieval, prompts, or ranking integration.
- **Answer feedback (outstanding)** should record whether an answer was useful and whether its cited evidence was relevant, linked to the query and source IDs for later evaluation. Current mismatch flags are session-only.

## Feature modules built on the shared foundation

Treat ingestion, retrieval, and provider access as shared infrastructure. Implement each user-facing capability as a small feature module with its own input/output contract and evaluation cases. A feature module may compose shared application services, but should not issue ad hoc SQL, call a vector database directly, or own a private copy of the provider integration. This keeps the features independently developable and lets improvements to retrieval and provenance benefit all of them.

| Feature module | Inputs and shared services | Output contract and behavior | Primary evaluation |
|---|---|---|---|
| **Review search and Q&A** | Natural-language query; typed review/diary/TMDb retrieval; structured SQLite lookups for ratings, dates, watched state, and other exact facts. | Answer only from supplied evidence. Return answer text plus source references (record/chunk ID, movie ID/title/year, source/content type, diary date, and excerpt). State when evidence is weak or absent; do not turn a review opinion into an objective movie fact. | Expected evidence retrieval, citation coverage, answer grounding, and empty/ambiguous query handling. |
| **Recommendation explanations** | Ranked result from the existing deterministic recommender; score components/reasons; retrieval for relevant personal reviews or highly rated comparison films. | Preserve ranker order and scores. Add a concise explanation with separate computed, personal-review, and TMDb evidence references. If retrieval finds nothing useful, retain the deterministic explanation. | Ranking unchanged, evidence relevance, attribution accuracy, and fallback when retrieval/provider is unavailable. |
| **Taste summaries and trend descriptions** | Aggregates computed by analysis services (rating distributions, time windows, entity affinity, genre/theme trends); retrieval for illustrative review excerpts. | Use computed results as the source of quantities and trends; use the model only to phrase/summarize them. Include time range, sample size, and references for examples; do not infer population-level claims from personal data. | Numeric fidelity to aggregates, temporal/filter correctness, evidence attribution, and no unsupported trends. |
| **Natural-language preference parsing** | User request plus supported recommender/filter schema (runtime, year/decade, genres, tone/theme, anchor, watchlist/discovery scope, and available score weights). | Return a validated, typed proposal with explicit filters/weights, unresolved terms, and a human-readable summary. The UI lets the user inspect/edit it before calling the deterministic ranker; parsing alone never mutates preferences or watchlist state. | Schema validity, intent-to-filter accuracy, ambiguity handling, and confirmation/read-only behavior. |
| **Filmweek personalization** | Existing Filmweek inputs and candidate validation; taste digest/aggregates; retrieval of relevant rated films and review evidence. | Keep the current curation contract and catalog verification. Add personal-evidence references to rationale where available; generated candidates remain subject to verification and never become library facts merely because the model returned them. | Candidate validity, evidence relevance, role diversity, and factual claim checks. |

### Later extension: film podcast transcripts

Podcast transcripts are a possible additional retrieval source, motivated by the user's interest in shows such as *The Big Picture* and *The Rewatchables*. Treat them as an optional feature built on the shared corpus and retrieval services, after the initial Letterboxd/TMDb path is working. They can provide outside criticism and discussion that may help explain or explore the user's movie interests, but they are not evidence of the user's own opinion or viewing history.

If this is pursued, ingest a small, deliberately selected sample first, ideally episodes about films already present in the Letterboxd library. Keep podcast material in its own source/content types, separate from `letterboxd_diary` and `tmdb`, and retain show, episode title, episode URL, publication date, speaker when available, and timestamps or segment boundaries. Link a segment to a canonical `movie_id` only when the film reference is clear; otherwise retain the episode-level record without inventing a movie link. Chunk around speaker turns or coherent discussion segments where practical, and preserve enough episode context to make retrieved excerpts understandable.

Retrieval and answer presentation must identify podcast discussion as attributed external commentary. Cite the show, episode, speaker, and timestamp/segment when available; do not blend it into personal reviews, treat it as an objective film fact, or infer that the user agrees with a host. Evaluate the sample with queries that distinguish personal-review evidence from podcast commentary, and check retrieval relevance and attribution before expanding the corpus. Keep a source link and respect the source's access terms; transcript availability, permitted reuse, and redistribution rights need to be assessed before building an importer or sharing transcript text. No podcast transcript importer or corpus schema is part of the current ingestion slice.

### Shared service contracts

- Define framework-neutral services (for example `ReviewRetriever`, `LibraryQueryService`, `RecommendationService`, `TasteAnalysisService`, and `LLMProvider`) under application/service modules. Streamlit is a caller, not the owner of business logic; evaluation and other clients use the same contracts.
- Retrieval accepts a query and bounded filters (source/content types, movie IDs, date/rating ranges, and result limit) and returns typed evidence records with stable source IDs, text, retrieval scores, and provenance. Support semantic, lexical, and hybrid modes without exposing backend-specific result shapes to feature modules.
- Structured facts come from SQLite/application services. Semantic similarity comes from retrieval. A combined request may call both and return separately typed evidence; an LLM must not guess missing structured facts.
- Define feature inputs/outputs as typed data (dataclasses or validated models), including source references and failure/empty states. Keep provider transport, prompt construction, parsing/validation, and feature policy in distinct layers.
- Keep feature modules callable without Streamlit and allow deterministic checks with fake retrievers, repositories, and providers. Record model/provider and prompt/schema versions for generated results where appropriate; do not retain raw prompts/reviews in logs by default.
- Feature modules may not call state-changing tools. Any later action (such as setting an anchor or adding a watchlist title) must be a separate explicit UI action with confirmation and its own application-service call.

### Remaining implementation order

1. Harden the initial **review search and Q&A** slice: cover structured intents, validate model citations against returned evidence, distinguish empty results from unavailable services, and verify provider/privacy behavior and injection resistance.
2. Add the fixed **RAG evaluation harness** and use it as a release gate for retrieval, generation, privacy, lifecycle, and degraded-mode changes.
3. Add persistent **answer and evidence feedback** linked to query and source IDs.
4. Add **recommendation explanations**, composing retrieval with the existing ranker without changing rank semantics.
5. Add **taste summaries** using deterministic aggregates first, then optional review excerpts for examples.
6. Add **natural-language preference parsing** as a constrained parser that proposes a validated request for user review before ranking.
7. Extend **Filmweek personalization** through the same retrieval and analysis contracts; retain its separate curation policy and catalog-validation path.
8. Add the safe full-index clear/rebuild operation and data-health reporting for imports, coverage, identity resolution, and index freshness; verify stale and deleted-source behavior through evaluation.

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

## Normalized retrieval corpus

`retrieval_corpus` is the SQLite staging contract for later chunking and indexing. It has one typed record per source event and content type. `record_id` is a deterministic SHA-256 key over `source`, `source_event_id`, and `content_type`; the same tuple is also unique. `movie_id` retains the canonical `lowercase title (year)` library key. `text` contains the searchable content, while `watched_date`, `rating`, and `rewatch` retain diary context. `tmdb_id` links external metadata records to TMDb. `created_at` is retained across upserts and `updated_at` changes when source content or context changes.

`retrieval_chunks` contains nonempty `review_text`, `tmdb_overview`, and `tmdb_genres` text. Short records remain intact; long records split on sentence/paragraph boundaries at a target of 200 model tokens, with 32-token overlap and an 8-token safety reserve under the model limit. Each chunk keeps the source record ID, stable chunk ID, chunk index/version, text hash, and serialized source provenance. `retrieval_embeddings` stores one active float32 unit vector per chunk, along with model ID/revision, vector dimension, source text hash, and timestamps. Unchanged text/model pairs reuse vectors; source edits refresh affected vectors; a model revision change rebuilds the active set. Foreign keys remove chunks and vectors when their corpus record is deleted. Diary context with empty text is retained in `retrieval_corpus` but is not embedded.

The default model is [`sentence-transformers/all-MiniLM-L6-v2` at pinned revision `680353e66602ffe5bfc6cea86f57a6458ad7e6dd`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/commit/680353e66602ffe5bfc6cea86f57a6458ad7e6dd). Install the optional dependency with `pip install -e '.[embeddings]'`, then run `python scripts/build_retrieval_index.py`. The script requires local model weights; it reports a setup error if the optional library or model is unavailable. Hosted embedding adapters are a future extension of the backend interface.

| Source | Content type | Text and context |
|---|---|---|
| `letterboxd_diary` | `diary_event` | Empty text; diary date, rating, and rewatch context for each export event. |
| `letterboxd_diary` | `review_text` | Personal review text with the same event ID and diary context. |
| `letterboxd_rss` | `diary_event` | Empty text; RSS event ID, date, rating, and rewatch context. RSS does not supply review text. |
| `tmdb` | `tmdb_overview` | TMDb plot overview and TMDb ID. |
| `tmdb` | `tmdb_genres` | Genre names joined as readable text and TMDb ID. |

Letterboxd event IDs use an event-specific identifier from the source row when available. Otherwise, import derives an ID from `movie_id`, watched date, and that event's occurrence index in the source export; mutable review text and rating are excluded. TMDb event IDs use the canonical movie key plus the metadata field, while `tmdb_id` records the external catalog identity. Imports upsert by source/event/content type, so unchanged imports do not add records and edits refresh the existing record. Blank review text removes only its `review_text` record while preserving the diary event. `load_retrieval_corpus()` exposes the active, joined records for downstream corpus building.

## Delivery sequence

1. **Complete:** Specify and validate the review corpus schema and idempotent import behavior.
2. **Complete:** Implement chunking and embedding refresh, first with a local development corpus.
3. **Complete:** Add pgvector setup, migrations/schema management, idempotent publishing, and provenance-aware semantic, lexical, and hybrid search. Verify repeat publish and edit/deletion propagation with focused and optional integration checks.
4. **Complete (initial slice):** Expose retrieval, recommendation, and SQLite library queries through framework-neutral typed services in `src/movie_recommender/application/`.
5. **Complete (initial slice):** Implement read-only routing, bounded retrieval, typed source references, provider/privacy policy, and basic degraded-mode reporting. Structured intent parsing and model citation validation remain incomplete.
6. **Complete (initial slice):** Add the Streamlit chat interface and evidence display. Conversation history and mismatch flags are session-only; persistent answer feedback and explicit follow-up action flows remain future work.
7. **Next:** Harden structured routing, citation validation, provider/privacy behavior, injection resistance, and empty/error handling in the initial chat path.
8. Add a fixed-query evaluation set with expected evidence; measure retrieval relevance, citation coverage, answer grounding, and ranking regressions separately. Run it as a release gate and use results to tune chunking and retrieval.
9. Add answer usefulness and evidence-relevance feedback, linked to the query and retrieved source IDs.
10. Add explicit user-triggered actions from evidence-backed answers, plus data-health reporting and a safe index clear/rebuild workflow.

Track implementation tasks in [`open-work.md`](open-work.md). Update the component table as each module is completed; keep the diagram and this document aligned when the target changes.
