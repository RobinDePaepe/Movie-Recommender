## Status legend
✅ Done · 🟡 Partial · 🔴 Not addressed · 🔵 Needs live verification

## Overall
- ✅ Score breakdown is a bit too black box still from within the interface, especially taste profiling and list signals
  - `render_score_breakdown()` (app.py) charts per-component contributions on Tonight's Pick + Recommendations, and now prints a plain-language sentence under each bar (`_component_explanations()`, app.py) — e.g. "List signals (+2.05): In 3 list(s): Top 250, Watchlist 2026" or "Taste similarity (+2.57): Matches your high-rated films on genres: ...". Sourced from data the pipeline already computed (`list_names`, `taste_matches`, `avg_user_rating`) but never surfaced.
- 🟡 Additionally it seems that list signals are too simplistic, just because a movie is in a couple of list does not mean that the movie should be a top candidate
  - Now weighted by list type (recommender.py:149) with count capped (clip(upper=5), recommender.py:321). Softer, but still a raw count×weight sum — not quality-aware.

- 🟡 Need to find a balance between discovering new movies and really evaluating watchlisted ones
  - Handled only via the "My watchlist" / "Not on my watchlist" mode radio + discovery tools. No blended/automatic balance.

- 🔴 I want to integrate watch counts into my app somewhere (Movies that I have watched a lot already should be taken into account somehow in my taste profiling)
  - Rewatch counts appear only as an Analysis metric; never feed recommender.py scoring.

- 🔴 Explore minimal but fun and additive LLM integration (manual triggers needed)
  - No LLM/OpenAI/Anthropic code anywhere in the project.

Your feedback:
1. ✅ Use the feedback system you already built. Wire the "Tune watched movies" UI from todo.md and start tagging real films. Until feedback has rows, half your scoring pipeline (add_feedback_similarity) is dead weight you can't tune.
   - Built on the Analysis page (app.py:881) with the full richer label set.
2. ✅ Establish a quality signal. Even something lightweight: pick 20 watched-but-"recommendable" films, see where the recommender ranks them, write the result into that empty testing-feedback.md. Right now you have no ground truth, so you can't know if a feature helped.
   - Evaluation page gives MAE / Precision@10 / NDCG holdout metrics — a real quality signal. The specific 20-film ranking exercise is now recorded below.


## Ground-truth ranking exercise (2026-07-10)
Method: sampled 20 films rated ≥4.0★ (seed 42), leave-N-out removed them from ratings/likes/watched/feedback so they don't leak into their own taste profile, dropped them into a candidate pool alongside your real 371-film watchlist (391 total), and ran the *actual production scoring pipeline* (`add_heuristic_scores` → `add_content_similarity` → `add_feedback_similarity` → `add_entity_affinity` → `add_theme_similarity`, Balanced taste mode, default weights, no anchor). Reports where each film landed by rank/percentile in that pool.

**Bug found + fixed during this run**: `theme_score` was silently `0.0` for every candidate, always. Root cause: your ratings have 345 films ≤2.5★ vs only 162 ≥4.0★, and `theme_similarity._scale_net()`'s fallback ("if nothing scores positive, return all zeros") triggered whenever the negative-penalty mean (over the much larger negative pool) outweighed the positive-similarity mean — which was true for nearly every candidate. Fixed by scaling on magnitude (`abs(values).max()`) instead of collapsing to zero when no positive reference exists (`theme_similarity.py::_scale_net`). All 28 tests still pass. The table below reflects the *fixed* pipeline.

| Rank | Percentile | Film | Your rating | Score | content | theme | entity | list |
|---|---|---|---|---|---|---|---|---|
| 2 | 99.7% | Before Sunrise (1995) | 4.5 | 5.02 | 2.57 | -1.14 | 1.07 | 2.05 |
| 4 | 99.2% | Inglourious Basterds (2009) | 5.0 | 2.74 | -1.45 | -1.31 | 0.29 | 4.99 |
| 9 | 97.9% | Django Unchained (2012) | 4.0 | 1.77 | -0.04 | -0.40 | 0.28 | 1.50 |
| 17 | 95.9% | Y Tu Mamá También (2001) | 4.5 | 1.07 | -0.72 | -0.94 | 0.15 | 2.37 |
| 21 | 94.9% | Lost in Translation (2003) | 4.5 | 0.55 | -1.84 | -1.50 | 0.55 | 3.12 |
| 25 | 93.8% | Dead Poets Society (1989) | 4.0 | 0.32 | -3.63 | -1.76 | 0.16 | 4.91 |
| 31 | 92.3% | Heat (1995) | 4.5 | 0.10 | -2.60 | -0.66 | -0.09 | 2.95 |
| 34 | 91.5% | Requiem for a Dream (2000) | 4.0 | -0.06 | -0.43 | -0.73 | 0.06 | 0.75 |
| 42 | 89.5% | The Wolf of Wall Street (2013) | 4.0 | -0.34 | -4.46 | -1.59 | 0.08 | 5.15 |
| 47 | 88.2% | Black Swan (2010) | 4.0 | -0.62 | -0.84 | -0.33 | 0.06 | 0.00 |
| 59 | 85.1% | 1917 (2019) | 5.0 | -0.88 | -3.97 | -2.05 | 0.38 | 4.15 |
| 79 | 80.0% | City of God (2002) | 4.0 | -1.48 | -3.35 | -2.09 | 0.11 | 3.61 |
| 171 | 56.4% | The Big Short (2015) | 4.0 | -3.26 | -4.32 | -1.08 | -0.05 | 1.65 |
| 176 | 55.1% | Moonlight (2016) | 4.0 | -3.35 | -2.77 | -1.91 | 0.00 | 0.75 |
| 226 | 42.3% | Carnage (2011) | 4.0 | -4.14 | -3.48 | -1.87 | 0.07 | 0.75 |
| 276 | 29.5% | 50/50 (2011) | 4.0 | -5.21 | -4.04 | -1.62 | 0.06 | 0.00 |
| 291 | 25.6% | The Phantom of the Opera at the Royal Albert Hall (2011) | 5.0 | -5.66 | -3.69 | -2.32 | -0.14 | 0.00 |
| 324 | 17.2% | Mrs. Doubtfire (1993) | 4.0 | -6.64 | -5.97 | -1.89 | 0.18 | 0.55 |
| 351 | 10.3% | The Lion King (1994) | 4.5 | -8.20 | -8.73 | -2.42 | 0.21 | 2.25 |
| 383 | 2.1% | Scott Pilgrim vs. the World (2010) | 4.0 | -12.22 | -9.34 | -2.90 | -0.46 | 0.00 |

**Aggregate**: 8/20 landed in the top 10% of the pool, 12/20 in the top 25%, median percentile 86.7%.

**Reading it**: the recommender clearly favors films with strong list signals (Inglourious Basterds, Dead Poets Society, The Wolf of Wall Street all carry big list-count boosts) over pure taste-similarity — several 4★+ films score *negative* on both `content` and `theme` despite being genuinely liked. That tracks with the still-open "list signals too simplistic" and "taste/list balance" items above. The bottom of the list (Scott Pilgrim, The Lion King, Mrs. Doubtfire) are outliers relative to your generally serious/prestige-drama-and-thriller-heavy positive set (Before Sunrise, Inglourious Basterds, Django) — i.e. the recommender is *not* wrong to rank them low relative to your dominant taste signal, even though you rated them highly; they're likely "guilty pleasure" style watches the `content`/`theme` channels can't distinguish from "doesn't fit your profile." Worth tagging films like these with the `guilty_pleasure` feedback label so `add_feedback_similarity` can compensate.

## Add a taste profile page
- 🔴 Really go in depth on how my taste is situated, how it has been developping
  - No dedicated taste-profile page and no taste-over-time view. Some pieces (rating distribution, avg-by-decade, genre taste table) live on the Analysis page, but nothing on the *evolution* of taste.


### Tonights Pick:
- 🔵 In the mood for does not have sufficient effect on the recommendations, there is too much overlap in recommendations between the movies recommended
  - Still just switches taste_mode; no code targets stronger pull or less overlap. Verify live.

### Analysis Page
- Im not happy yet with the Tune watched movies page yet a couple of reasons:
        - ✅ Visualisation of the movies is still a bit clunky and too random, movies are now order alphabetical with no way to edit the list execpt for the search featur
          - Now has search, "only untagged", rating-range filter, signal-based sort (untagged → rating desc), pagination, and poster thumbnails.
        - ✅ Taste feedback options still do not cover how I want to evaluate a movie
          - FEEDBACK_LABELS expanded to 16 nuanced labels (recommender.py:40).
        - 🔴 Adding review data (If already available from letterboxd) for future LLM usage would be a fun addityion
        - 🔴 Seeing the effects of the feedback would make the tuning watched movie feature more interactive
        - 🔴 The graphs are visually jarring (coloring, too basic of a setup both from a visual as well as substantive level), Considering if these visuals should live separate from the tune watched movies function ( and maybe move it to a "My taste profile overview"-page)

### Curated Weeks:
- 🟢 The selection of movies seems too random still and genre overlap seems to be too dominant of a factor (f.e. Mission Impossible recommendations when selecting Inception as a movie anchor)
  - theme_similarity.py + per-style theme_similarity weight + Theme-focused style directly target genre-dominance. Curation is now "what a film is about" aware. Subjective quality still needs your eyeball.


## Still to tackle (net)
1. 🔴 Watch-count integration into taste profiling
2. 🔴 Minimal manual LLM integration
3. 🔴 Dedicated taste-profile / taste-evolution page (+ move the jarring graphs there, restyle them)
4. 🔵 Stronger "In the mood for" effect on Tonight's Pick
5. 🔴 Pull Letterboxd review text (LLM groundwork)
6. 🔴 Show feedback effects live on the Tune page
7. ✅ Richer plain-language transparency for taste/list scores
8. ✅ Record the 20-film ground-truth ranking results
9. 🔴 (new, found during #8) List signals appear to outweigh taste/theme similarity for several genuinely-liked films — worth revisiting list weighting vs. content/theme weighting balance
