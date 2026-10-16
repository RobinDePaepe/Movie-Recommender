# Personal Movie Discovery

A local Streamlit app for exploring a Letterboxd film library, finding films to watch, and understanding your taste. It combines your Letterboxd export and recent Letterboxd activity with TMDb metadata, configurable recommendation signals, curated film weeks, and optional LLM-assisted discovery.

## Features

- **Personal recommendations:** rank films from your watchlist or discover candidates outside it. Scoring can combine decade and recency preferences, Letterboxd list membership, content and thematic similarity, director/writer/cast affinity, an optional anchor film, and saved feedback.
- **Explanations and tuning:** inspect recommendation reasons and score components, adjust signal weights, and give positive or negative feedback. Feedback can also be applied while tuning recommendations against films you have already watched.
- **Curated film weeks:** build an ordered programme around an anchor film, with narrative roles and styles such as balanced, director-focused, theme-focused, or vibe-focused.
- **Filmweek discovery:** optionally use an LLM provider to generate discovery picks informed by a digest of your taste. Provider packages are included in `requirements.txt`; configure the key for the provider you select.
- **Taste and rating analysis:** explore ratings over time, entity affinity, feedback and reflection patterns, list affinity, and comparisons with mainstream ratings.
- **Film reflections:** record category-level judgements and evidence about a film to help understand and refine your preferences.
- **Metadata enrichment:** use TMDb for genres, credits, keywords, descriptions, recommendations, and similar films. OMDb can provide additional ratings where configured.
- **Letterboxd sync:** optionally fetch recent diary and rating activity from Letterboxd RSS and apply it as an incremental overlay.

## Requirements

- Python 3.9 or newer
- Python packages declared through `pyproject.toml` and `requirements.txt`
- A Letterboxd export ZIP for your library
- A TMDb API key for metadata enrichment

Optional packages and credentials:

- `sentence-transformers` enables local embedding-based thematic similarity. Without it, thematic similarity falls back to TF-IDF.
- `statsmodels` enables a trendline on the evaluation chart.
- LLM provider credentials are needed only when using the corresponding Filmweek backend. See `.env.example` for supported environment variable names.
- Letterboxd RSS sync can use `LETTERBOXD_USERNAME` or `LETTERBOXD_RSS_CLIENT_ID` (which may be a direct RSS feed URL).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -e .
```

Set credentials in your shell or create a `.env` file in the project root. At minimum, set the TMDb key:

```dotenv
TMDB_API_KEY=your_tmdb_api_key
LETTERBOXD_USERNAME=your_letterboxd_username
```

See `.env.example` for optional service keys and settings. Do not commit `.env` or personal exports.

Put the standard Letterboxd export ZIP at `data/letterboxd_export.zip`. The `data/` directory is ignored by Git because it contains personal exports, local databases, caches, and sync files.

## Run the app

```bash
streamlit run app.py
```

The app loads its library from the SQLite database when one exists; otherwise it can use the Letterboxd export and file-based metadata. Use the database controls in the app to build or rebuild the SQLite library after importing new data. If RSS sync credentials are configured, the app can sync recent activity at startup (at most once per session when the last sync is over an hour old).

Use the sidebar to manage metadata enrichment, recommendation settings, discovery modes, and analysis. TMDb enrichment can also be run from the command line:

```bash
python scripts/enrich_tmdb.py --limit 100
```

To sync Letterboxd RSS manually and inspect sync status:

```bash
python scripts/sync_letterboxd.py YOUR_LETTERBOXD_USERNAME --status
```

## Architecture

- `app.py` — thin Streamlit launcher for `src/movie_recommender/ui/app.py`.
- `src/movie_recommender/recommendations/` — candidate selection, scoring, explanations, feedback, and thematic similarity. Theme similarity uses sentence-transformer embeddings when installed and TF-IDF otherwise.
- `src/movie_recommender/curation/` — role-based film-week curation and LLM-generated discovery.
- `src/movie_recommender/analysis/` — taste analysis and film reflection logic.
- `src/movie_recommender/data/` — SQLite schema and access, Letterboxd imports, ratings, and RSS sync.
- `src/movie_recommender/integrations/` — TMDb, OMDb, and LLM provider clients.
- `src/movie_recommender/ui/` — Streamlit UI and reflection panel.
- `scripts/` — command-line tools for database setup, TMDb enrichment, and Letterboxd sync.
- `tests/` — tests for recommendation, curation, database, sync, theme similarity, and reflection behavior.

### Recommendation flow

Letterboxd export → normalized library data → optional RSS overlays → candidate selection → scoring and explanations → Streamlit results. When SQLite is present, the app reads library data from the database. The shared movie key is `movie_id`, formatted as `lowercase title (year)`.

The recommender combines several independently weighted signals. Content similarity uses genres, directors, cast, keywords, and overview; thematic similarity uses keywords and overview, separate from genre and credits. Feedback and mood avoidance can adjust results, and an anchor film can focus the recommendation toward a specific concept. The controls and scoring constants are documented in `CLAUDE.md` and `AGENT.md`.

## Development

Run the test suite with:

```bash
pytest tests/
```

Run one test with:

```bash
pytest tests/test_recommender.py::test_build_recommendations_no_metadata_returns_results
```

See `AGENT.md` for repository-specific development commands, architecture details, and data-handling guidance. `CLAUDE.md` contains the corresponding Claude Code notes.

The current architecture target is the [RAG design](docs/rag-architecture.md), based on [RAG architecture.png](docs/RAG%20architecture.png). Track active implementation work in [docs/open-work.md](docs/open-work.md); superseded plans and status reports are preserved in `archive/`.

## Local data and privacy

The app stores its SQLite database, Letterboxd exports, TMDb cache, theme embeddings, feedback, and sync overlays under `data/`. These files are ignored by Git. Keep API keys in environment variables or `.env`; never add personal exports or credentials to commits.
