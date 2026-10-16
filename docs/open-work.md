# Open Work

Last reviewed: 2026-10-16.

The main product and engineering goal is to build the modules needed to realize the [RAG architecture](rag-architecture.md). This file is the active backlog. Older notes and detailed historical results are in `../archive/`.

## Priority 1 — Complete the RAG path

1. **Review ingestion and normalization**
   - Preserve Letterboxd review text, diary context, and stable movie/event identifiers during import.
   - Normalize imported reviews and TMDb plot/genre metadata into a documented retrieval corpus.
   - Make import and re-import idempotent; track source and source-event identifiers so updates do not create duplicate chunks.
2. **Review chunking and embeddings**
   - Define chunk boundaries and metadata for reviews, including movie, date, rating, and source identifiers where available.
   - Generate and refresh embeddings when source text changes; record model/version information.
3. **pgvector retrieval store**
   - Add a pgvector-backed store for review and metadata chunks, with a documented local setup and rebuild path.
   - Define how the store relates to the existing SQLite library and how IDs keep the two stores in sync.
   - Combine semantic vector search with PostgreSQL full-text search for exact titles, people, and phrases as well as conceptual matches; evaluate Reciprocal Rank Fusion or another simple result-fusion method.
4. **Retrieval and agent orchestration**
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

## Priority 2 — Data quality and recommendation follow-up

- Resolve the outstanding movie identity review queue and measure remaining unresolved records.
- Add a data-health view for last import/sync times, unresolved identities, review and metadata coverage, and retrieval-index freshness, with clear next steps for stale or missing data.
- Complete review-text ingestion and expose it in the library workflow. Diary review text is already retained when present in the export, but it is not yet normalized into a searchable review corpus.
- Revisit the balance between Letterboxd list membership and personal taste signals using a checked-in ranking regression set.
- Calibrate mood modes, Tonight's Pick diversity, and watchlist-versus-discovery balance against observed use.
- Add persistent user preferences for scoring weights, variety, default taste mode, and filters.
- Improve score explanations to identify the specific films, entities, lists, and feedback events behind each component.

## Priority 3 — Product experience

- Build a dedicated Taste Profile experience for taste composition and change over time, and improve the visual hierarchy of existing analysis views.
- Improve Curated Weeks interaction: pin, replace, reorder, and complete slots; then refine emotional pacing and repetition control.
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
