# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the app:**
```bash
.\.venv\Scripts\Activate.ps1   # Windows (PowerShell)
streamlit run app.py
```

**Install dependencies:**
```bash
pip install -r requirements.txt
pip install statsmodels  # optional: enables trendline on evaluation chart
```

**Run tests:**
```bash
pytest tests/
pytest tests/test_recommender.py::test_build_recommendations_no_metadata_returns_results  # single test
```

**Enrich TMDb metadata from CLI:**
```bash
python enrich_tmdb.py --limit 100
python sync_letterboxd.py $LETTERBOXD_USERNAME --status
```

## Environment

The app reads `TMDB_API_KEY` and `LETTERBOXD_USERNAME` from the environment or a `.env` file. The Letterboxd export zip must be at `data/letterboxd_export.zip`.

## Architecture

The app is a single-page Streamlit UI (`app.py`) that delegates all logic to these modules:

**`recommender.py`** — the core engine. The scoring pipeline is:
1. `candidate_pool()` — picks watchlist or outside-watchlist candidates
2. `add_heuristic_scores()` — decade affinity, liked-decade bonus, recency, list signals (`LIST_SCORE_SCALE`, `LIST_COUNT_WEIGHT` constants control list weight)
3. `add_content_similarity()` — TF-IDF cosine similarity over `feature_text` (genres×4, directors×5, cast×2, keywords×3, overview) against high-rated films; negative-rated films subtract a penalty
4. `add_feedback_similarity()` — TF-IDF similarity to films you've tagged with `FEEDBACK_LABELS` feedback
5. `add_entity_affinity()` — Bayesian-average rating per director/writer/cast entity
6. `add_anchor_similarity()` — optional boost toward one user-picked anchor film
7. `apply_mood_avoidance()` — session-only mood penalty
8. Final `score` = weighted sum of all components; weights are user-tunable in the sidebar

**`curator.py`** — builds ordered "curated weeks" around an anchor film. Assigns narrative roles (`Context / influence`, `Thematic setup`, `Anchor movie`, etc.) to slots, then scores and picks candidates per role using `STYLE_WEIGHTS` (Balanced, Director-focused, Theme-focused, Vibe-focused, etc.).

**`movie_database.py`** — SQLite backend (`data/movie_recommender.sqlite`). The app auto-detects whether the DB exists and uses it instead of CSV/JSON files. The DB must be rebuilt via the sidebar button after new imports. All tables are defined in `init_db()`. JSON list columns (genres, cast, etc.) are stored as JSON strings and deserialized on load.

**`tmdb_client.py`** — `TMDbClient` wraps the TMDb API. `enrich_movies()` batch-fetches and caches results to `data/tmdb_cache.json`. `discover_movies_from_favorites()` finds new outside-watchlist candidates via TMDb's recommendations/similar endpoints.

**`letterboxd_sync.py`** — incremental RSS sync for recent diary/rating activity, writing overlays to `data/sync/`. RSS is additive; a fresh export remains the source of truth for watchlist state and old rating edits.

**Data flow:** Letterboxd export (zip) → `load_letterboxd()` → optional RSS overlay → `candidate_pool()` → scoring pipeline → Streamlit display. With the SQLite backend active, `load_data_from_db()` replaces the CSV step.

**`movie_id` key:** `"{name lower} ({year})"` — used everywhere as the join key across all frames.
