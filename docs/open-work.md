# Open Work

Last reviewed: 2026-10-19.

The main product and engineering goal is to build the modules needed to realize the [RAG architecture](rag-architecture.md). This file is the active backlog. Older notes and detailed historical results are in `../archive/`.

## Priority 1 — Complete the RAG path

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
4. **Application services and agent orchestration (next)**
   - Define framework-neutral application services around the retrieval repository and existing SQLite library/recommender functions.
   - Route exact library facts and filters to SQLite/application services, semantic opinion/theme questions to vector retrieval, and mixed questions to both.
   - Add a small orchestrator that selects from narrow, documented retrieval and existing recommender/analysis tools.
   - Keep deterministic ranking authoritative for ranked recommendations; use retrieved evidence to ground answers and explanations.
   - Return source film/review references with answers so results can be inspected.
   - Treat imported reviews and metadata as untrusted evidence, never as instructions. Keep tools read-only by default and test prompt-injection attempts in retrieved text.
   - Make provider and data-sharing behavior explicit: disclose when review text leaves the device, avoid sending personal text to hosted models without a user-configured provider/workflow, and define chat-history retention and clearing behavior.
   - Preserve clear source types in answers so personal opinions, TMDb facts, and computed recommendation results cannot be conflated.
5. **Chat interface**
   - Add a natural-language query interface for questions over the user's reviews and movie library.
   - Show supporting films/reviews and handle empty or weak retrieval results clearly.
   - For each cited item, show film title/year, review date or metadata source, a short excerpt, and a way to open the source record; let the user flag mismatched evidence.
   - Remain useful when the LLM provider or vector store is unavailable: fall back to structured library search and deterministic recommendations, and label which evidence path produced the result.
   - Add explicit user-triggered actions from answers, such as adding a film to the watchlist, using it as a recommendation anchor, or starting a curated week. Keep the initial query and answer tools read-only.
6. **Evaluation harness**
   - Build a fixed set of representative questions with expected evidence and repeatable retrieval/answer checks.
   - Report retrieval relevance, citation coverage, answer grounding, and recommendation ranking quality separately; retain the existing rating holdout metrics as one input, not the only measure.
   - Make the evaluation set a release gate for changes to chunking, embedding models, retrieval, prompts, and ranking integration.
   - Include source attribution, privacy/provider routing, stale or deleted source handling, prompt-injection resistance, and degraded-mode behavior in the evaluation checklist.
7. **Answer feedback loop**
   - Once the core chat flow is usable, let the user mark answer usefulness and evidence relevance.
   - Store feedback with query and retrieved source identifiers so it can inform later evaluation and retrieval tuning.

## Feature modules after shared retrieval

Develop these as separate modules over the shared application services and typed evidence contract in [the RAG architecture](rag-architecture.md#feature-modules-built-on-the-shared-foundation). Each module should have its own interface, failure behavior, and evaluation cases; do not duplicate ingestion, retrieval, provider setup, or source attribution.

1. **Review search and Q&A** — retrieve personal review evidence with provenance, combine with structured library facts when needed, and return cited answers with clear weak/no-evidence behavior.
2. **Recommendation explanations** — enrich existing ranked results with relevant review or comparison-film evidence while preserving rank order and scores; retain deterministic explanations as fallback.
3. **Taste summaries and trend descriptions** — compute counts and trends in analysis code, then use the LLM only to describe those results and select sourced examples.
4. **Natural-language preference parsing** — parse user language into a validated recommender request, show the proposal for inspection/editing, and only then call the deterministic recommender.
5. **Filmweek personalization** — connect the existing curator to shared taste aggregates and retrieval evidence while retaining candidate verification and the separate curation contract.

Suggested sequence: add framework-neutral application services and orchestration over the completed retrieval API, then ship review search/Q&A as the proving feature, followed by recommendation explanations, taste summaries, preference parsing, and Filmweek personalization. Later modules should keep the shared service boundaries and evidence rules.

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
- Propagate source edits and deletions through normalized corpus records, chunks, embeddings, and retrieval caches; provide a safe full-index clear/rebuild operation and verify that removed review text is no longer retrievable.
- Cache reusable data loading and scoring across Streamlit reruns where it measurably improves responsiveness.
- Keep service boundaries independent of Streamlit so the orchestrator and evaluation harness can use the same import, retrieval, recommendation, and analysis functions.

## Conditional work

- A web API is only needed if a second client or hosted integration is introduced.
- Multi-user database hosting is only needed if shared use or concurrent writes become a requirement. The RAG diagram's pgvector store is the vector retrieval component; it does not by itself require replacing SQLite as the library/scoring data store.
