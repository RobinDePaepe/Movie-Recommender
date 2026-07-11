# Feedback & Development Status

Consolidated from `todo.md`, `movie_curation_todo.md`, and `testing-feedback.md`.
Status verified against the codebase on 2026-07-05.

**Legend:** ✅ Done · 🟡 Partial · 🔴 Open · ⚪ Irrelevant / superseded

---

## 1. Watched-Movie Feedback Tuning (from `todo.md`)

**Goal:** Use already-watched movies as explicit taste signals, especially for the "More or Less like this" evaluation flow.

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Richer feedback constants in `recommender.py` | ✅ Done | `FEEDBACK_LABELS` ([recommender.py:39](recommender.py:39)) covers all 7 requested labels plus extras (`masterpiece`, `great_film`); back-compat `more_like_this` / `less_like_this` keys preserved. |
| 2 | Update feedback weight map in `add_feedback_similarity` | ✅ Done | Weights sourced from `FEEDBACK_LABELS` ([recommender.py:517](recommender.py:517)); negative weights retained for "less like this" style labels; watched-tuning feedback carries its own scope. |
| 3 | "Tune watched movies" UI block in `app.py` | ✅ Done | Analysis page ([app.py:881](app.py:881)): search, rating-range filter, "only untagged", pagination, posters, multiselect, per-page save. |
| 4 | Optional `feedback_scope` / `note` columns | ✅ Done (2026-07-05) | `scope` column + migration ([movie_database.py:165](movie_database.py:165)); tagging writes `scope="watched_tuning"`. Free-text notes now added: `movie_notes` table + `save_movie_note()` / `load_movie_notes()`, surfaced as a **"Taste note"** field per film in the Tune watched movies UI (DB mode), saved for future reference and LLM use. 3 unit tests. |

**Acceptance criteria**
- ✅ Watched movies can be marked with richer taste labels.
- ✅ Recommendation scoring changes based on those labels.
- ✅ Existing `more_like_this` / `less_like_this` feedback still works.
- ✅ App no longer treats all high ratings as automatically "more like this."

---

## 2. Curated Weeks (from `movie_curation_todo.md`)

**Goal:** Build an ordered watchlist ("Curated Weeks") around one anchor movie.

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | `curator.py` module (`build_curated_list`, `total_movies` default 7) | ✅ Done | Module present; roles assigned per slot. |
| 2 | Streamlit "Curated Weeks" page | ✅ Done | [app.py:1046](app.py:1046). |
| 3 | Controls (anchor source, 3–14 slider, style selector, checkboxes) | ✅ Done | Anchor from watched/rated/watchlist; include-anchor / allow-watched / allow-watchlisted checkboxes. |
| 4 | Render curated result (day, role, title, year, why, poster, genres/moods) | ✅ Done | Includes an intensity / "flow across the week" line chart ([app.py:1142](app.py:1142)). |
| 5 | Export functionality (CSV, then Markdown/JSON/Letterboxd) | ✅ Done (2026-07-05) | Curated Weeks now offers **Download as CSV** and **Download as Markdown** (shareable "movie week card" via `curated_week_to_markdown()`), both UTF-8, filename slugged from the anchor. JSON/Letterboxd-import formats remain future ideas. |
| 6 | Optional database support (save / re-open weeks) | ✅ Done | `save_curated_week` / `load_curated_week`; tables in place. |
| 7 | Future improvements (regenerate slot, pin, drag-drop, advanced modes, LLM text, mood/decade graphs) | 🔴 Open | Nice-to-haves, untouched. |

---

## 3. Testing Feedback (from `testing-feedback.md`)

### Overall

| Item | Status | Notes |
|------|--------|-------|
| Score breakdown too black-box (esp. taste profiling & list signals) | 🟡 Partial | A `why` column and mood-penalty breakdown exist, but taste-profiling / list-signal transparency is not surfaced in the UI. |
| List signals too simplistic (presence in a few lists shouldn't top-rank a film) | ✅ Done | Reworked into `aggregate_list_signal()` with geometric corroboration decay — strongest list counts in full, extras only corroborate. |
| Balance discovery vs. evaluating watchlisted films | 🟡 Partial | Addressed via `taste_mode` / candidate pool selection; not fully tuned. |
| Integrate watch counts into taste profiling | 🔴 Open | No `watch_count` / rewatch signal feeds scoring in `recommender.py`. |
| Explore minimal, additive LLM integration (manual triggers) | 🔴 Open | No LLM code anywhere in the repo (`anthropic` / `openai` absent). |

### Reviewer's suggested next steps
| Item | Status | Notes |
|------|--------|-------|
| Wire the "Tune watched movies" UI and start tagging real films | ✅ Done | UI shipped (see §1.3); tagging persists via feedback flow. |
| Establish a quality signal / ground truth (rank ~20 known films, record results) | 🔴 Open | No ground-truth set recorded; `testing-feedback.md` still holds no ranked results. |

### Taste profile page
| Item | Status | Notes |
|------|--------|-------|
| Dedicated in-depth "taste profile" page (how taste is situated & has developed) | 🟡 Partial | Only a basic "Genre taste profile" table inside Analysis ([app.py:875](app.py:875)); no dedicated page and no over-time development view. |

### Tonight's Pick
| Item | Status | Notes |
|------|--------|-------|
| "In the mood for" has too little effect; too much overlap between picks | 🔴 Open | Mood filter/avoidance exists, but effect strength / overlap not visibly tuned. |

### Analysis page — "Tune watched movies"
| Item | Status | Notes |
|------|--------|-------|
| Movie visualisation clunky / alphabetical / only searchable | ✅ Done | Now search + rating filter + signal-based sorting + pagination + posters. |
| Taste feedback options don't cover how I want to evaluate a movie | ✅ Done | Richer `FEEDBACK_LABELS` set (masterpiece, rewatchable, guilty pleasure, high-quality-not-my-taste, etc.). |
| Add Letterboxd review data (for future LLM use) | 🔴 Open | Not imported / displayed. |
| See the effect of feedback (make tuning interactive) | 🔴 Open | No live feedback-effect view. |
| Graphs visually jarring; consider moving to a "My taste profile overview" page | 🔴 Open | Still default Plotly styling inside Analysis; not relocated. |

### Curated Weeks
| Item | Status | Notes |
|------|--------|-------|
| Selection too random; genre overlap dominates (e.g. Inception → Mission Impossible) | ✅ Done | Added `theme_similarity.py` engine; styles carry a semantic `theme_similarity` weight isolated from genre/director/cast. |

---

## 4. Superseded / no longer relevant

| Item | Why |
|------|-----|
| Two-label back-compat concern (`todo.md` §1) | Superseded by the full `FEEDBACK_LABELS` map, which already preserves the old keys. |
| `feedback_scope` framed as "optional later, not required" (`todo.md` §4) | Overtaken by events — the `scope` column now exists. Only the free-text `note` idea remains. |
| "List signals too simplistic" critique (`testing-feedback.md`) | Resolved by the corroboration-decay rework; no longer an open concern. |

---

## 5. Top open items (recommended focus)

1. **Watch-count taste signal** — feed rewatch/watch counts into taste profiling.
2. **LLM integration** — minimal, manually-triggered (natural-language explanations, curated-week intros).
3. **Dedicated "My Taste Profile" page** — in-depth + development-over-time, with polished visuals.
4. **Curated Weeks export** — CSV first, then Markdown / Letterboxd import format.
5. **Ground-truth quality signal** — rank ~20 known films and record results to enable regression testing.
6. **"In the mood for" tuning** — stronger effect and less overlap on Tonight's Pick.

---

## 6. Newly identified issues & additions (code audit, 2026-07-05)

Surfaced from a codebase audit — not present in the original three feedback files.

### 🔴 Highest impact — technical

| # | Item | Status | Notes |
|---|------|--------|-------|
| A | **No caching — scoring pipeline re-runs on every interaction** | ✅ Done (2026-07-05) | Added `@st.cache_data` wrappers in `app.py`: `load_db_bundle` / `load_csv_bundle` for data loading and `cached_recommendations` for scoring, keyed on a `data_token()` of backing-file mtimes so writes (feedback, RSS sync, DB rebuild, new export/cache) invalidate the cache while UI interactions hit it. Measured **~15× faster reruns** (2.0s → 0.13s) with unchanged data; all 33 tests pass and an AppTest smoke run is exception-free. |
| B | **No diversity / novelty re-ranking (root cause of "too much overlap")** | ✅ Done (2026-07-05) | Added `diversity_rerank()` (MMR over `feature_text` cosine similarity) in `recommender.py`, wired through `cached_recommendations` and controlled by a new **"Variety"** sidebar slider (default 0.35). Keeps the top-scored pick fixed, then spreads franchise/director/genre clusters apart across the list (and across Tonight's Pick "give me another"). `feature_text` is carried through build output only for this step and dropped before display/CSV. 4 unit tests added; verified on real data (top pick preserved, noir cluster de-clustered). |

### 🟡 Correctness & robustness

| # | Item | Status | Notes |
|---|------|--------|-------|
| C | **Test coverage is engine-only** | ✅ Done (2026-07-05) | Added **49 tests** across 4 new files: `test_letterboxd_sync.py` (16), `test_movie_database.py` (12), `test_curator.py` (11), `test_theme_similarity.py` (10). Suite now **86 passing**. Theme tests run on the TF-IDF fallback so no embedding lib is required. |
| — | **Bug found while writing tests** | ✅ Fixed (2026-07-05) | `load_curated_week()` called `pd.read_json(json_string)`, which pandas 3 treats as a *file path* — "Load a saved curated week" crashed with `FileNotFoundError`. Wrapped the payload in `io.StringIO` ([movie_database.py](movie_database.py)). |
| D | **RSS-includes-lists bug has no regression test** | ✅ Done (2026-07-05) | `test_parse_rss_items_filters_out_list_entries` guards the list-filter ([letterboxd_sync.py:112](letterboxd_sync.py:112)); plus a test that a numeric title (`1917`) never invents a star rating. |
| E | **Broad `except Exception` blocks, several silent** | ✅ Done (2026-07-05) | Audited all sites. Added a module logger and logging to the ones that could mask real failures: startup **auto-sync** now logs a warning and surfaces "⚠️ Auto-sync skipped: …" in the sidebar ([app.py](app.py)); theme embedding-cache read/write and model-load, and TMDb enrich/details network failures now log ([theme_similarity.py](theme_similarity.py), [tmdb_client.py](tmdb_client.py)). Benign type-coercion fallbacks (`_safe_year`, `_decade`, date parsing) intentionally left silent. |

### 🟢 Product additions (not in current feedback)

| # | Item | Status | Notes |
|---|------|--------|-------|
| F | **Persist user settings** | 🔴 Open | Scoring weights, taste mode, and mood filters live only in widget state and reset each session. Saving to the DB makes tuning stick. |
| G | **Explainability panel** | 🔴 Open | Directly answers the "score breakdown is a black box" note: per-recommendation expander showing top contributing films/entities/lists behind each component. |
| H | **Cold-start / onboarding + empty states** | 🔴 Open | App assumes a populated library; a first-run user with no cache/ratings hits bare `st.info` messages. |
| I | **Watchlist prioritization view** | 🔴 Open | "Watchlist aging / why haven't I watched this yet" — a concrete feature for the discovery-vs-watchlist balance goal. |

### ⚪ Minor housekeeping

| # | Item | Status | Notes |
|---|------|--------|-------|
| J | Duplicate commit | ⚪ Housekeeping | `9103584` and `0c68b37` share an identical message ("Scope outside-watchlist pool…") — likely an accidental double-commit. |
| K | `venv/` vs `.venv/` mismatch | ⚪ Housekeeping | CLAUDE.md / README tell users `.venv`, but the repo has a `venv/` folder. |
| L | Security check | ✅ Clean | `.env` is gitignored and untracked; API key is not committed. |

### Suggested order
Caching (**A**) and the diversity re-ranker (**B**) give the most user-visible improvement; the RSS regression test (**D**) is cheap insurance against a bug already hit once. — **A and B done; D next.**
