# Repository guidance for coding agents

`AGENT.md` is the source of truth for repository commands, architecture, and development guidance. Keep this file current when the project changes. `CLAUDE.md` points Claude Code to this file and should not duplicate its contents.

## Project layout

- `app.py` — thin Streamlit launcher.
- `src/movie_recommender/ui/` — Streamlit app and reflection panel.
- `src/movie_recommender/recommendations/` — candidate selection, scoring, feedback, and thematic similarity.
- `src/movie_recommender/curation/` — role-based film-week curation and LLM-assisted discovery orchestration.
- `src/movie_recommender/analysis/` — taste analysis and reflection logic.
- `src/movie_recommender/data/` — SQLite database access, Letterboxd sync/import, and external rating retrieval.
- `src/movie_recommender/integrations/` — TMDb, OMDb, and LLM provider clients.
- `scripts/` — command-line utilities for database setup, metadata enrichment, and Letterboxd sync.
- `tests/` — automated tests, organized by feature.
- `docs/` — design notes, plans, and historical project documentation.

The project uses a `src/` package layout. Install it in editable mode before running the app or tests so imports resolve consistently.

## Setup and commands

Create and activate a virtual environment, then install the project and its dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -e .
```

Run the app:

```bash
streamlit run app.py
```

Run tests:

```bash
pytest tests/
pytest tests/test_recommender.py::test_build_recommendations_no_metadata_returns_results
```

Run the command-line tools:

```bash
python scripts/init_database.py --status
python scripts/enrich_tmdb.py --limit 100
python scripts/sync_letterboxd.py YOUR_LETTERBOXD_USERNAME --status
```

Optional packages:

- `sentence-transformers` enables semantic theme embeddings; without it, theme similarity uses the TF-IDF fallback.
- `statsmodels` enables a trendline on the evaluation chart.

## Environment and local data

The app reads `TMDB_API_KEY` and `LETTERBOXD_USERNAME` (or `LETTERBOXD_RSS_CLIENT_ID`, which may be a direct RSS feed URL) from environment variables or a root `.env` file. LLM provider credentials and their variable names are listed in `.env.example`.

Place the Letterboxd export at `data/letterboxd_export.zip`. The `data/` directory contains local exports, databases, caches, feedback, and sync overlays and is ignored by Git. Do not commit credentials or personal Letterboxd data.

## Application behavior and architecture

The Streamlit UI lives in `src/movie_recommender/ui/app.py`. It coordinates the feature modules and starts an RSS sync at most once per session when Letterboxd credentials are configured and the last sync is over an hour old. In database mode, synced overlays are applied to SQLite. Streamlit theme defaults are in `.streamlit/config.toml`.

The recommendation pipeline in `recommendations/recommender.py` selects candidates, computes weighted scores, and builds explanations. Its main signals include:

1. Heuristic preferences for decade, recency, and Letterboxd list membership.
2. Content similarity using genres, directors, cast, keywords, and overview.
3. Feedback similarity and director, writer, and cast affinity.
4. Optional anchor similarity and thematic similarity. Theme similarity uses keywords and overview, with local sentence-transformer embeddings when available and a TF-IDF fallback otherwise.
5. Session-level mood avoidance.

The theme engine is shared by recommendations and film-week curation. The `curation/` package builds ordered programmes around an anchor film and can use configurable LLM providers for discovery. `data/movie_database.py` owns SQLite access and schema setup; `data/letterboxd_sync.py` stores incremental RSS activity under `data/sync/`. A fresh Letterboxd export remains the source for complete watchlist state and older rating edits.

The shared `movie_id` join key is `"{name lower} ({year})"`. Preserve its format when changing imports, sync, or joins.
