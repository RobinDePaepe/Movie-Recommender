# Personal Movie Recommender MVP

This repository is a small personal movie recommender that combines your Letterboxd export with TMDb metadata to produce watch recommendations and explanations.

**What it does**
- Ranks unwatched items using Letterboxd signals (custom lists, decade affinity, recency) plus a TMDb-based content-similarity score (TF‑IDF on genres/directors/cast/keywords/overview).
- Lets you give feedback (`more_like_this` / `less_like_this`) that adjusts scores.
- Provides a lightweight evaluation page that tests how well similarity predicts past ratings.

Prerequisites
- Python 3.9+ (use a virtual environment)
- Recommended packages listed in `requirements.txt`.
- Optional: install `statsmodels` to enable a trendline on the evaluation chart: `pip install statsmodels`.

Quick setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

TMDb API key
- To enrich metadata and enable better content-similarity, set your TMDb API key in the environment or create a `.env` file containing:

```
TMDB_API_KEY=your_api_key_here
```

Usage

- Place your Letterboxd export file at `data/letterboxd_export.zip` (the app expects the standard Letterboxd export structure).
- Start the Streamlit app:

```bash
streamlit run app.py
```

- Use the sidebar to fetch TMDb metadata (small batches recommended initially), toggle recommendation source (watchlist / outside-watchlist), and view evaluation.

Files of interest
- `recommender.py` — scoring, content-similarity, explainers, and recommendation builder.
- `tmdb_client.py` / `enrich_tmdb.py` — fetching and caching TMDb metadata.
- `app.py` — Streamlit UI.

UI notes
- The recommendations table shows a concise `why` summary; click a movie in the details pane to see the full `why_details` explanation and matched lists/taste matches.
- The evaluation page will fall back to a plain scatter plot if `statsmodels` is not installed.

Customization
- To reduce/increase the influence of list-based signals, edit the constants in `recommender.py`:
  - `LIST_SCORE_SCALE` — scales per-list scoring contribution
  - `LIST_COUNT_WEIGHT` — weight for number of lists the movie appears on

Git / Data
- The repository includes a `.gitignore` that ignores `data/` and common non-essential files (`*.csv`, `*.env`, Excel files, archives, virtualenvs, caches, IDE folders).
- If you want to keep an empty `data/` folder in the repo, add a `data/.gitkeep` file.

Troubleshooting
- If the app fails to show TMDb metadata, run the enrich step again and check `data/tmdb_cache.json`.
- If evaluation errors reference `statsmodels`, install it as noted above.

Questions or next steps
- I can add a short developer guide with tests and example commands, or create `data/.gitkeep` for you — which would you prefer?