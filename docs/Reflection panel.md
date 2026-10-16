# Reflection Panel — Pilot Plan

## Goal
A single new Streamlit page that takes one film (rated, watchlisted, or new) and surfaces facet-level evidence from existing signals, so you can form your own rating/review — no scoring, no persistence in v1.

## Scope for pilot (cut everything else)
- **In scope:** movie resolution (3 cases), 3 facets only (director, cast, theme), evidence display, free-text rating/review inputs, session-only state.
- **Out of scope for v1:** writer affinity, content-similarity facet, drift-check for re-rating, review-tone scaffold, `reflections` table/persistence. Add these in v2 once the core loop feels useful — no point building all 5 facets before confirming the display format is actually useful to you.

Reasoning for the cut: director + cast are your richest entity-affinity data (most films have both), theme is the signal you explicitly want isolated and are most curious about. Writer affinity is often sparse (missing/uncredited a lot in TMDb) and content-similarity is the one facet that's closest to "re-deriving the recommender," so it's the safest one to defer.

## Step 1 — `resolve_for_reflection(movie_id_or_query)`
Before writing this, paste the current interfaces for:
- `movie_database.py`: however you currently look up a film by `movie_id` (function name, return shape)
- `tmdb_client.py`: the search/fetch function used for new titles
- however `feature_text` is currently constructed in `recommender.py` (so the new-film path builds an identical structure)

Function should return one consistent shape regardless of source:
```
{
  movie_id, title, year, genres, director, cast (list), keywords, overview,
  in_db: bool, rating: float | None
}
```
Case handling:
- in DB + rated → pull all fields + rating
- in DB + watchlisted → pull all fields, rating=None
- not in DB → TMDb search/fetch, build feature fields to match `recommender.py`'s convention, rating=None, nothing written anywhere

## Step 2 — Facet evidence functions
Each facet is a **query against existing signal machinery**, not a new computation. Confirm these exist before implementing — paste signatures if unsure:
- Entity affinity lookup: does `recommender.py` expose a function like `entity_affinity(name, role)` you can call directly, or is it only computed inline during scoring? If inline-only, this is the one piece of real refactor work in the pilot: extract it into a standalone callable.
- Theme similarity: `theme_similarity.py` should already expose something like `most_similar(movie_id, candidates, k)` — confirm the candidate set can be "all my rated films" rather than only watchlist/candidate_pool.

For each of director / cast / theme, produce:
```
{facet: "director", value: "<name>", history: [(title, year, rating), ...], n: int}
```
`n` matters — if `n == 0`, the panel should show "no prior data" rather than an empty table. Don't let a zero-evidence facet render identically to a real one.

## Step 3 — Page layout (`rate_review.py`)
Rough structure, top to bottom:
1. Film header (title, year, poster if easy via TMDb, genres)
2. If `in_db and rating is not None`: small badge showing existing rating (no drift-check logic yet, just visible context)
3. Three facet panels (director / cast / theme), each showing name + evidence table + `n`
4. Free-text rating input (number/slider) + free-text review box
5. No save button in v1 — this is scratch space, refresh clears it. Confirms the "not persisted" requirement cheaply, and tells you fast whether the panel format is actually useful before you invest in a `reflections` table.

## Step 4 — Entry point
Simplest for pilot: a text input / selectbox at the top of the page where you paste or pick a `movie_id`, rather than wiring it into existing watchlist/recommendation cards yet. Wiring it into other pages is a v2 nicety once the page itself proves useful — don't couple the two pieces of work.

## What to paste before I can help implement
1. `movie_database.py` — the lookup-by-`movie_id` function
2. `recommender.py` — however entity affinity is currently computed (inline in scoring loop, or standalone function)
3. `theme_similarity.py` — its public function signatures
4. `tmdb_client.py` — the search/fetch function for a title not yet in your DB

## Known weak spots to watch for once built
- New/obscure films will often have thin or missing cast/keyword data from TMDb — expect "no prior data" to show up more than you'd like on the cast facet specifically.
- If entity affinity is currently only computed inline during scoring (not a standalone callable), extracting it is the one non-trivial refactor in this plan — worth confirming before you start rather than discovering it mid-implementation.