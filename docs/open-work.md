# Open Work

Last reviewed: 2026-10-19.

The main product and engineering goal is to build the modules needed to realize the [RAG architecture](rag-architecture.md). This file is the active backlog. Older notes and detailed historical results are in `../archive/`.

## Priority 1 — Make the initial RAG path reliable

1. **Review ingestion and normalization (implemented)**
   - Preserve Letterboxd review text, diary context, and stable movie/event identifiers during import.
   - Normalize imported reviews and TMDb plot/genre metadata into a documented retrieval corpus.
   - Make import and re-import idempotent; track source and source-event identifiers so updates do not create duplicate chunks.
2. **Review chunking and embeddings (implemented)**
   - Define chunk boundaries and metadata for reviews, including movie, date, rating, and source identifiers where available.
   - Generate and refresh embeddings when source text changes; record model/version information.
3. **pgvector retrieval store (implemented and locally checked)**
   - Versioned PostgreSQL migration and Docker Compose setup for pgvector.
   - Idempotent publisher mirrors active SQLite chunks and embeddings, updating edits and removing deleted chunks.
   - `PgVectorRepository` provides semantic, lexical, and reciprocal-rank-fused hybrid search with provenance and structured filters.
   - Focused publisher/search tests pass; the optional PostgreSQL lifecycle test covers repeat publishing, evidence provenance, edits, and deletions in an isolated schema.
4. **Application services and orchestration (initial slice implemented; hardening remains)**
   - Framework-neutral typed adapters cover SQLite library lookups, pgvector retrieval, the existing deterministic ranker, and deterministic taste analyses (`src/movie_recommender/application/`).
   - `ChatOrchestrator` routes structured, semantic, mixed, recommendation, and analysis requests; returns typed source references and labels evidence paths.
   - Recommendation output preserves ranker order. Retrieval/provider failures are reported and semantic retrieval falls back to a structured library lookup.
   - Query answering is read-only and stateless. The optional provider contract is opt-in; hosted providers do not receive personal library facts or Letterboxd excerpts unless `allow_hosted_personal_data=True`. Retrieved text is delimited and treated as untrusted evidence in the prompt.
   - The route classifier is intentionally small and heuristic. Callers can supply validated `LibraryQuery`, retrieval filters, and recommendation requests, but free-text parsing does not yet cover all intents the router recognizes (for example, count questions).
   - Remaining: validate generated citations against returned source IDs; strengthen structured request parsing and ambiguous-query handling; test privacy/provider routing, prompt injection, errors, and fallbacks. The implementation has no RAG evaluation release gate yet.
5. **Chat interface (implemented, initial slice)**
   - The Chat page calls the framework-neutral orchestrator and services; it shows answer, evidence path, provider disclosure, and citation cards with title/year, source/date, excerpt, and a link to Letterboxd or TMDb.
   - Empty evidence and unavailable services get explicit messages. Exact library answers remain deterministic; recent-watch queries use dated diary events and return in watch-date order. Users can constrain retrieval by title/source and flag mismatched evidence for the current session.
   - Provider selection is explicit, defaults to off, and requires a separate opt-in before a hosted provider receives personal library evidence. Conversation history stays in session state and has a Clear chat control.
   - Query and answer tools remain read-only. Mismatch flags and conversation history are session-only. Persistent evidence feedback, richer weak-evidence scoring, and separate follow-up actions remain future work.
6. **Evaluation harness (not implemented for RAG)**
   - Build a fixed set of representative questions with expected evidence and repeatable retrieval/answer checks.
   - Report retrieval relevance, citation coverage, answer grounding, and recommendation ranking quality separately; retain the existing rating holdout metrics as one input, not the only measure.
   - Make the evaluation set a release gate for changes to chunking, embedding models, retrieval, prompts, and ranking integration.
   - Include source attribution, privacy/provider routing, stale or deleted source handling, prompt-injection resistance, and degraded-mode behavior in the evaluation checklist.
7. **Answer feedback loop (not implemented)**
   - Let the user mark answer usefulness and evidence relevance.
   - Store feedback with query and retrieved source identifiers so it can inform later evaluation and retrieval tuning.

## Feature modules after shared retrieval

Develop these as separate modules over the shared application services and typed evidence contract in [the RAG architecture](rag-architecture.md#feature-modules-built-on-the-shared-foundation). The current chat is an initial review Q&A slice; it still needs the reliability and evaluation work above. Each later module should have its own interface, failure behavior, and evaluation cases; do not duplicate ingestion, retrieval, provider setup, or source attribution.

1. **Review search and Q&A hardening** — validate citations and structured intents, test weak/no-evidence and service failure behavior, and make the initial chat slice reliable before expanding its scope.
2. **Recommendation explanations** — enrich existing ranked results with relevant review or comparison-film evidence while preserving rank order and scores; retain deterministic explanations as fallback.
3. **Taste summaries and trend descriptions** — compute counts and trends in analysis code, then use the LLM only to describe those results and select sourced examples.
4. **Natural-language preference parsing** — parse user language into a validated recommender request, show the proposal for inspection/editing, and only then call the deterministic recommender.
5. **Filmweek personalization** — connect the existing curator to shared taste aggregates and retrieval evidence while retaining candidate verification and the separate curation contract.

Suggested sequence: (1) harden the current review search/Q&A slice and its provider/privacy boundaries; (2) add the fixed RAG evaluation set and make it a release gate; (3) persist answer/evidence feedback; then (4) build recommendation explanations, taste summaries, preference parsing, and Filmweek personalization. Add data-health reporting and safe index rebuild/clear operations alongside this work. Keep later modules on the shared service boundaries and evidence rules.

## Priority 2 — Data quality and recommendation follow-up

- Resolve the outstanding movie identity review queue and measure remaining unresolved records.
- Add a data-health view for last import/sync times, unresolved identities, review and metadata coverage, and retrieval-index freshness, with clear next steps for stale or missing data.
- Expose normalized review text and diary context in the library workflow for inspection.
- Revisit the balance between Letterboxd list membership and personal taste signals using a checked-in ranking regression set.
- Calibrate mood modes, Tonight's Pick diversity, and watchlist-versus-discovery balance against observed use.
- Add persistent user preferences for scoring weights, variety, default taste mode, and filters.
- Improve score explanations to identify the specific films, entities, lists, and feedback events behind each component.

## Priority 3 — Product experience

- Build a dedicated Taste Profile experience for taste composition and change over time, and improve the visual hierarchy of existing analysis views.
- Improve Curated Weeks interaction: pin, replace, reorder, and complete slots; then refine emotional pacing and repetition control.
- Validate every AI-discovery pick against a film catalog (title, release year, and director) before presenting it as a factual film; clearly label or replace unverified candidates when catalog verification is unavailable. See the *Whiplash* / Ollama reproduction in `../archive/testing-feedback.md`.
- Add watchlist aging using age, intent, availability, and prompts about why a title remains unwatched.
- Improve onboarding and empty states so setup progress and missing data are clear.

## Foundation work to schedule with the RAG modules

- Add versioned database migrations and typed validation for imported data, settings, and tool requests.
- Add structured operational logs and backups around mutating import, sync, and enrichment workflows.
- Source edits and deletions already propagate through SQLite corpus records, chunks, embeddings, and pgvector publishing. Still provide a safe full-index clear/rebuild operation and include deleted-source checks in the evaluation gate.
- Cache reusable data loading and scoring across Streamlit reruns where it measurably improves responsiveness.
- Application services and orchestration are framework-neutral; keep the evaluation harness on those same interfaces and avoid moving retrieval/business logic into Streamlit.

## Conditional work

- A web API is only needed if a second client or hosted integration is introduced.
- Multi-user database hosting is only needed if shared use or concurrent writes become a requirement. The RAG diagram's pgvector store is the vector retrieval component; it does not by itself require replacing SQLite as the library/scoring data store.
