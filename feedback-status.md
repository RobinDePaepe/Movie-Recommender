# Development Status and Roadmap

Status verified against the codebase on 2026-09-16.

Legend: ✅ done · 🟡 partial/in progress · 🔴 open · ⚪ deferred

## Current foundation

| Area | Status | Current state |
|---|---|---|
| Letterboxd ingestion | ✅ | Full export plus incremental RSS overlays; list RSS entries and numeric-title rating errors are guarded by tests. |
| Metadata | ✅ | TMDb enrichment, movie/TV identity review, media typing, posters, cast, crew, keywords, runtime and release data. |
| Recommendation engine | 🟡 | Hybrid list, content, theme, entity, feedback, decade, recency and watchlist-intent scoring exists; general diversity re-ranking and objective calibration remain open. |
| Tonight's Pick | ✅ | Runtime, mood, media-type and availability controls plus skip/feedback interaction. |
| Availability and ownership | 🟡 | Structured region/provider/format/subtitle/expiry data and manual filtering exist; automated provider refresh is open. |
| Taste feedback | ✅ | Rich feedback labels, watched-tuning scope, notes and similarity-based score adjustments. |
| Reflection and watch context | ✅ | Category reflections plus per-viewing mood, companions, setting, note and rewatch intent. |
| Profiles and watchlist intent | 🟡 | Profiles, intent and per-profile preferences exist; joint scoring blends average/minimum fit and applies strong-dislike vetoes. Real preference data and calibration are still needed. |
| Curated Weeks | 🟡 | Deterministic and LLM-assisted curation, saved weeks and LLM-pick regeneration exist; dedicated exports and richer editing remain open. |
| Analysis | 🟡 | Ratings, entity affinity, feedback, reflections and mainstream comparison exist; a cohesive taste-profile experience is open. |
| Testing | ✅ | Unit coverage across recommendation, database, sync, theme, reflection, curator and LLM curation flows. |

## Active goal: improve personal taste signals

This is the current development focus.

1. ✅ **Rewatch-aware taste scoring** — reliable diary counts and explicit rewatch flags now produce a bounded affinity channel; bulk “marked watched” rows are excluded.
2. 🟡 **Mood calibration** — taste modes now use stronger positive/negative evidence, runtime rules and visible match reasons; real-world calibration is still needed.
3. ✅ **Visible feedback impact** — saved/removed labels now produce a before/after score and rank-impact panel.
4. 🟡 **Profile-aware taste** — per-profile ratings, positive signals, notes and vetoes now feed a 70% average / 30% least-satisfied blend. The remaining work is entering real preferences and calibrating the policy from observed results.

## Next product goals

| Priority | Goal | Status | Definition of done |
|---|---|---|---|
| 1 | Recommendation ground truth | 🔴 | A checked-in set of known positives/negatives and ranking regression tests with useful quality metrics. |
| 2 | Complete personal taste signals | 🟡 | Rewatch signal, stronger mood modes, feedback-effect UI and profile-aware joint scoring are implemented; real-world calibration remains. |
| 3 | Library-data completion | 🟡 | Resolve the identity queue, import Letterboxd reviews and add a compliant availability-provider refresh path. |
| 4 | My Taste Profile | 🔴 | A dedicated, visually coherent page showing taste composition and change over time. |
| 5 | Persistent preferences | 🔴 | Scoring weights, variety, default taste mode and filters survive a new session. |
| 6 | Deeper explainability | 🟡 | Explanations name the specific films, entities, lists and feedback events responsible for each score. |
| 7 | Watchlist aging | 🟡 | A prioritized view using age, intent, availability and “why haven't I watched this?” prompts. |
| 8 | Curated Weeks interaction | 🟡 | Pin, replace, reorder and complete slots; improve emotional pacing and repetition control. |
| 9 | Onboarding | 🔴 | Guided first run, useful empty states and clear setup progress. |

## Technical roadmap

| Horizon | Goal | Status |
|---|---|---|
| Now | Versioned database migrations and typed validation | 🔴 |
| Now | Ranking regression suite and import idempotency coverage | 🟡 |
| Soon | Separate import/enrichment/recommendation services from Streamlit | 🔴 |
| Soon | Structured operational logs and automatic backups around every mutation | 🟡 |
| Soon | Cache data loading and recommendation scoring across Streamlit reruns | 🔴 |
| Later | Add semantic embeddings alongside TF-IDF and compare them objectively | 🔴 |
| Conditional | FastAPI only when another client is required | ⚪ |
| Conditional | PostgreSQL only for hosted multi-user/concurrent use | ⚪ |

## Completed or superseded items

- Rich watched-movie feedback labels, feedback scope and taste notes.
- Bounded list scoring so stacked list membership cannot grow without limit.
- Theme similarity isolated from genre/director/cast similarity.
- LLM-assisted Filmweek curation with multiple providers and verified TMDb facts.
- Curated Week saving/reopening and LLM-pick regeneration.
- Availability/ownership schema, media typing, identity quarantine and watch-date correction.
- External IMDb and Letterboxd aggregate-rating enrichment.

`todo.md` and `movie_curation_todo.md` retain the detailed history of their original feature work. This file is the canonical current roadmap.
