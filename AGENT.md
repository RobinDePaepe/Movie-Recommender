# Repository guidance for coding agents

This file adapts the repository's `CLAUDE.md` guidance for coding agents working on this project. Keep both files aligned when changing shared commands, environment variables, or architecture.

## Commands

**Run the app:**

```bash
streamlit run app.py
```

**Install dependencies:**

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
pip install statsmodels            # optional: evaluation chart trendline
pip install sentence-transformers  # optional: semantic theme similarity
```

**Run tests:**

```bash
pytest tests/
pytest tests/test_recommender.py::test_build_recommendations_no_metadata_returns_results
```

**Enrich TMDb metadata from the command line:**

```bash
python enrich_tmdb.py --limit 100
```

**Sync Letterboxd RSS and inspect status:**

```bash
python sync_letterboxd.py YOUR_LETTERBOXD_USERNAME --status
```

## Environment and local data

The app reads `TMDB_API_KEY` and `LETTERBOXD_USERNAME` (or `LETTERBOXD_RSS_CLIENT_ID`, which may be a direct RSS feed URL) from the environment or a `.env` file. LLM provider keys are described in `.env.example` and are only needed for the provider in use.

Place the standard Letterboxd export ZIP at `data/letterboxd_export.zip`. Local data, exports, databases, caches, and sync overlays are ignored by Git. Do not commit credentials or personal Letterboxd data.

## Architecture

The app is a single-page Streamlit UI (`app.py`) that delegates business logic to modules. It injects a custom dark-cinema theme (`inject_theme()`, `render_hero()`, `score_badge_html()`, `chips_html()`). Startup auto-sync runs once per session when `LETTERBOXD_USERNAME` or `LETTERBOXD_RSS_CLIENT_ID` is set and the last sync was more than an hour ago. It pulls RSS activity, applies overlays to SQLite when the database is active, and persists the username to `.env`. Streamlit theme defaults live in `.streamlit/config.toml`.

### Recommendation pipeline (`recommender.py`)

1. `candidate_pool()` selects watchlist or outside-watchlist candidates.
2. `add_heuristic_scores()` scores decade affinity (Bayesian-shrunk toward the global mean via `DECADE_PRIOR_COUNT`), liked-decade bonus, recency (linear decay from `0.8` in the current year to `0` over a 15-year window), and list signals. `LIST_SCORE_SCALE` and `LIST_COUNT_WEIGHT` control list influence.
3. `add_content_similarity()` calculates TF-IDF cosine similarity over `feature_text` (genres ×4, directors ×5, cast ×2, keywords ×3, and overview) against highly rated films. Low-rated films contribute a penalty.
4. `add_feedback_similarity()` compares candidates with films tagged using `FEEDBACK_LABELS`.
5. `add_entity_affinity()` calculates Bayesian-average ratings for director, writer, and cast entities.
6. `add_anchor_similarity()` optionally boosts similarity to a user-selected anchor, using the theme engine and a scale of `4.0`.
7. `add_theme_similarity()` adds a separate conceptual similarity score (`theme_score`) based on keywords and overview from highly rated films, excluding genre and credit features.
8. `apply_mood_avoidance()` applies a session-only mood penalty.
9. The final `score` is a weighted sum. Sidebar weights control content, theme, entity, list, and anchor signals. Heuristics are damped tie-breakers: only deviation from the `3.0` floor contributes, scaled by `HEURISTIC_WEIGHT`. Anchor-focus (on by default) scales content, theme, and entity weights by `ANCHOR_FOCUS_SCALE` while an anchor is active.

### Key modules

- `theme_similarity.py` — shared thematic similarity engine. `theme_text()` uses keywords and overview only. It uses local `all-MiniLM-L6-v2` sentence-transformer embeddings, cached in `data/theme_embeddings.pkl` and invalidated by SHA-1 text hashes. If the optional package is unavailable, it falls back to TF-IDF. Provides `theme_anchor_scores()` and `theme_taste_scores()`, both scaled to roughly 0–4.
- `curator.py` — builds ordered, role-based curated weeks around an anchor. Narrative roles include context/influence, thematic setup, and the anchor. Styles such as Balanced, Director-focused, Theme-focused, and Vibe-focused adjust `STYLE_WEIGHTS`, including the thematic similarity contribution.
- `llm_curator.py` / `llm_providers.py` — Filmweek picks generated through configurable LLM backends.
- `movie_database.py` — SQLite backend at `data/movie_recommender.sqlite`. Schema is created by `init_db()`. The app prefers the database when it exists; rebuild it from the export and metadata cache after new imports. JSON list columns are stored as JSON strings and decoded on load. After `sync_rss()`, call `apply_rss_overlays_to_db()` in DB mode to upsert rating history and deduplicated diary events without a full rebuild.
- `tmdb_client.py` — `TMDbClient` wraps TMDb. `enrich_movies()` fetches and caches metadata in `data/tmdb_cache.json`; `discover_movies_from_favorites()` finds outside-watchlist candidates using TMDb recommendations and similar-film endpoints.
- `letterboxd_sync.py` — incremental RSS sync for recent diary and rating activity, stored as overlays under `data/sync/`. RSS is additive; the export remains the source of truth for watchlist state and older rating edits.
- `analysis.py`, `reflection.py`, and `rate_review.py` — analysis shaping helpers and reflection workflows.
- `app.py` — Streamlit UI and application orchestration.

### Data flow and identity

Letterboxd export → `load_letterboxd()` → optional RSS overlay → `candidate_pool()` → scoring → Streamlit display. In database mode, `load_data_from_db()` supplies the library in place of CSV loading.

The shared `movie_id` join key is `"{name lower} ({year})"`. Preserve this format when changing imports, sync, or joins.
