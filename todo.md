# Watched-Movie Feedback Tuning

Status verified on 2026-09-16. The original feature is complete; remaining work has moved into the broader personal-taste goal.

## Completed

- Rich feedback labels, including positive, neutral and negative taste distinctions.
- Backwards compatibility with `more_like_this` and `less_like_this`.
- Stronger weighting for deliberate watched-movie tuning.
- Searchable/filterable Tune Watched Movies interface.
- Feedback scope and free-text taste notes.
- Similarity-based propagation from tagged films into recommendation scores.

## Remaining extensions

- Import Letterboxd review text for analysis and optional LLM context.

## Added on 2026-09-16

- Reliable diary counts and explicit rewatch flags feed a bounded rewatch-affinity score.
- Bulk Letterboxd “marked watched” events remain excluded from count-based taste scoring.
- Taste modes use stronger positive/negative evidence and expose their matched evidence.
- Feedback changes display before/after score and rank movement.
- Per-profile ratings, positive signals and vetoes feed a blended joint score.

See `feedback-status.md` for the canonical prioritized roadmap.
