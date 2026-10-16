# Add a thematic / conceptual similarity dimension

## Context

The recommender's notion of "similar to Inception" is really "same director / genre / cast,"
because every signal is collapsed into one TF-IDF bag (`_feature_text`) dominated by
directors×5, genres×4, cast×2, and TF-IDF further rewards rare director/cast tokens. The
"aboutness" of a film (dreams, nested realities, puzzle-box structure) survives only as sparse
`keywords` and a diluted `overview`, so it is drowned out. Two concrete symptoms, both visible
in the user's own Inception-anchored export:

- The anchor only meaningfully boosted same-director/genre films (The Odyssey 4.0, Dune 1.31),
  not concept-cousins.
- **Moon (anchor 0.51) and Primer (anchor 0.32)** — the most Inception-like candidates — sank
  to the bottom because `content_score` (global taste) is **added** to the anchor and cancelled
  it (Moon content −1.05, final −1.42).

**Goal:** introduce a dedicated *theme/concept* similarity dimension that (a) measures what a
film is *about* using semantic embeddings of `keywords + overview` only, (b) drives a
theme-aware anchor and a standalone "Theme similarity" taste slider, (c) feeds the curator's
Theme-focused style, and (d) stops global taste from cancelling an active anchor via an
"anchor-focus" mode. So anchoring on Inception surfaces concept-cousins (Moon, Primer, Source
Code-likes), and the engine can express "films about the same idea," not just "by the same author."

## Decisions (locked in with the user)

- **Theme engine:** local semantic embeddings (`sentence-transformers`, `all-MiniLM-L6-v2`),
  cached to disk, with an automatic **TF-IDF fallback** when the library isn't installed (so
  the app and tests never hard-depend on it).
- **Apply to (broadest):** theme-aware anchor **and** a standalone theme slider **and** the
  curator's Theme-focused style.
- **Combination:** **anchor-focus mode** — when a film is anchored, automatically attenuate the
  global-taste signals so the anchor leads. Normal (no-anchor) browsing is unchanged.

## Architecture overview

New module **`theme_similarity.py`** owns: theme-text extraction, the embedding backend +
on-disk cache, the TF-IDF fallback, and the similarity helpers. `recommender.py` and
`curator.py` both call it, so the recommender and curated weeks share one definition of "theme."

```
theme_text(row)            -> "keywords ... \n overview"   (NO genre/director/cast)
get_model()                -> cached SentenceTransformer or None (lazy, optional import)
embed_movies(meta)         -> {movie_id: np.float32[384]}, disk-cached, hash-invalidated
theme_anchor_scores(...)   -> Series 0..4  (candidate vs anchor theme vector)
theme_taste_scores(...)    -> Series 0..4  (candidate vs rating-weighted high-rated theme set)
# fallback: same signatures using TfidfVectorizer over theme_text when get_model() is None
```

## Detailed changes by file

### 1. `theme_similarity.py` (new)

- **`theme_text(row)`** — join `keywords` (via existing `recommender._as_list`) + `overview`.
  Deliberately excludes genre/director/cast so the channel is pure concept. Empty when both missing.
- **Embedding backend**
  - `get_model()`: lazy module-global singleton; `try: from sentence_transformers import SentenceTransformer` else return `None`. Model `all-MiniLM-L6-v2` (384-dim, ~80MB).
  - `embed_movies(meta)`: for each `movie_id`, compute `sha1(theme_text)`; reuse cached vector if the hash matches, else batch-embed the misses. Persist to **`data/theme_embeddings.pkl`** as `{movie_id: {"hash": str, "vec": np.float32}}`. ~1500 films ≈ a few MB; first run a few seconds on CPU, cache hits after.
- **Similarity helpers** (cosine via `sklearn.metrics.pairwise.cosine_similarity`, already a dep)
  - `theme_anchor_scores(candidates, anchor_movie_id, meta)`: cosine of each candidate's theme vector vs the anchor's, then robust-scaled to 0–4 (see Normalization).
  - `theme_taste_scores(candidates, ratings, likes, meta)`: mirror `add_content_similarity`'s positive/negative selection (recommender.py:323–329) and rating weighting (`_rating_weight`, recommender.py:309–311), but in theme-embedding space; rating-weighted mean cosine to the positive set minus an optional negative penalty; robust-scaled to 0–4.
- **TF-IDF fallback** (when `get_model()` is None): build a `TfidfVectorizer(min_df=1, ngram_range=(1,2), max_features=12000)` over `theme_text` only and compute the same two scores — same shape as curator's `_similarity_to_anchor` (curator.py:220–234), so behavior is familiar and tested.

### 2. `recommender.py`

- **Theme-aware anchor:** rewrite `add_anchor_similarity` (528–554) to delegate to
  `theme_similarity.theme_anchor_scores` (embeddings, fallback to theme-only TF-IDF). Keep the
  output column name **`anchor_score`** to minimize plumbing — it just becomes concept-based.
- **Standalone theme channel:** add `add_theme_similarity(candidates, ratings, likes, metadata)`
  producing a **`theme_score`** column via `theme_similarity.theme_taste_scores`. Call it in
  `build_recommendations` right after `add_content_similarity` (≈ line 584). Reuse the existing
  positive/negative id logic rather than duplicating it.
- **Score assembly** (597–606): add `theme_w = float(weights.get("theme", 1.0))` and a
  `+ candidates["theme_score"] * theme_w` term.
- **Anchor-focus mode:** new constant `ANCHOR_FOCUS_SCALE = 0.4` and param
  `anchor_focus: bool = True`. When `anchor_movie_id` is set and `anchor_focus`, scale the
  personal-taste weights — `content_w`, `theme_w`, `entity_w` — by `ANCHOR_FOCUS_SCALE` before
  the sum (leave `list_w`, heuristic, and `anchor_w` at full). This is what rescues Moon/Primer.
- **Output + explanations:** add `"theme_score"` to the `cols` list (614); extend
  `explain_short` (473–497) and `explain_detailed` (500–525) with a theme reason (e.g.
  `theme_score >= 2.0 → "Thematically similar"`), optionally naming shared keywords via curator's
  `_overlap` for richer detail.

### 3. `curator.py`

- Add a `theme_similarity` key to every entry in `STYLE_WEIGHTS` (48–115), highest in
  **Theme-focused** (71–81).
- In `_score_candidates` (237–286): compute an embedding theme-similarity Series for the pool vs
  the anchor (reuse `theme_similarity.theme_anchor_scores`) and add
  `theme_sim * weights["theme_similarity"] * <scale>` to the `score` (263–273). Keep the existing
  keyword `_overlap` so the "shared themes" reason text (296) is unchanged.

### 4. `app.py`

- **Sidebar** (429–435): add `theme_weight = st.slider("Theme similarity", 0.0, 3.0, 1.0, 0.25, help="How strongly conceptual/thematic similarity (what a film is *about*) affects the score.")` and `"theme": theme_weight` in `score_weights`.
- **Anchor block** (680–701): add an `st.checkbox("Anchor focus", value=True, help="Let the anchored film lead by easing off your general taste profile.")`; thread it as `anchor_focus=` into the Recommendations re-call (711–714).
- **Score breakdown** (`render_score_breakdown`, 528–573): read `theme_w`; add
  `("Theme similarity", theme_score * theme_w)` to `components`; when an anchor is active+focused,
  apply `ANCHOR_FOCUS_SCALE` to the displayed Taste/Theme/Dir-Cast rows so the chart stays honest
  (pass an `anchor_active` flag in).
- **Table** (`show_cols`, 749): add `"theme_score"`. CSV (775) picks it up automatically.

### 5. `requirements.txt` + `CLAUDE.md`

- Keep core deps as-is. Add an **optional** install line/comment for `sentence-transformers`
  (note it pulls in `torch`). Document in CLAUDE.md: `pip install sentence-transformers` enables
  semantic theme matching; without it the app falls back to TF-IDF theme similarity. Add a one-line
  pipeline note for the new theme step.

## Normalization fix (applies to anchor + theme scores)

Replace the single-outlier `sims / max_sim * 4.0` (recommender.py:372, 553) with robust scaling
in the new helpers: scale by a high **percentile** of the candidate sims (e.g. 95th) and clip to
[0, 4]; when the candidate set is tiny (< ~8, as in unit tests), fall back to max-scaling so
existing monotonic assertions hold. This stops one same-director outlier from flattening genuine
concept-cousins toward zero. Leave `content_score`'s own normalization untouched to avoid
disturbing current behavior/tests.

## Tests (`tests/test_recommender.py`)

Fallback runs automatically in CI/local (no embedding lib required), so tests target the
TF-IDF path and pass without new deps.

- Extend `_metadata_data()` (20–46) with `overview` text and a **concept-cousin** film: a
  thematically matching candidate with a *different* director/genre/cast (so `content_score`
  wouldn't surface it but `theme_score`/anchor should).
- `test_theme_weight_raises_thematic_candidate`: mirror `test_anchor_weight_raises_similar_candidate` (233–242) — with `score_weights={"theme":0}` vs `{"theme":3}`, the cousin's score rises.
- `test_anchor_focus_rescues_concept_cousin`: a high-anchor / negative-taste film ranks higher with `anchor_focus=True` than `False` (the Moon/Primer scenario).
- Add a `theme_similarity` smoke test: `theme_taste_scores`/`theme_anchor_scores` return finite 0–4 Series on the fixture with the lib absent.
- Confirm existing suite still passes (esp. the anchor test — keep `anchor_score` monotonic).

## Verification

1. `pytest tests/` — all green via the TF-IDF fallback.
2. (Optional, real engine) `pip install sentence-transformers`, then `streamlit run app.py`.
3. **Inception sanity check** (the original complaint): Recommendations → "Not on my watchlist"
   or watchlist, Anchor on **Inception (2010)**, Anchor focus ON, Theme slider ≥ 1. Confirm
   concept-cousins (Moon, Primer, and similar) climb the list and Sofia-Coppola-style dramas no
   longer dominate; open "Score breakdown" to see the new Theme/Anchor bars carrying weight.
4. Curated Weeks → anchor Inception, style **Theme-focused**; confirm the week leans on concept
   overlap and "shared themes" reasons.
5. First run writes `data/theme_embeddings.pkl`; a second run is fast (cache hits).

## Risks / notes

- **Dependency weight:** `torch` is large. Mitigated by lazy import + automatic TF-IDF fallback;
  nothing breaks if it's never installed.
- **Per-rerun cost:** embeddings are cache hits and cosine over ~1500×384 is trivial; the model
  loads once per process (module-global singleton).
- **Cache staleness:** keyed by `sha1(theme_text)`, so re-enriched metadata re-embeds only the
  changed films. Safe to delete `data/theme_embeddings.pkl` to force a rebuild.
- **Behavior change:** anchor-focus only activates when an anchor is set; default browsing
  rankings are unaffected apart from the new (default-1.0) theme term.

## Suggested implementation order

1. `theme_similarity.py` (theme_text, embedding backend + cache, fallback, two score helpers).
2. Wire `recommender.py` (theme-aware anchor, `add_theme_similarity`, score sum, anchor-focus, cols, explanations).
3. Tests for steps 1–2; iterate to green.
4. `app.py` UI (slider, anchor-focus checkbox, breakdown row, table column).
5. `curator.py` theme weight + scoring.
6. `requirements.txt` / `CLAUDE.md`; manual Inception verification.