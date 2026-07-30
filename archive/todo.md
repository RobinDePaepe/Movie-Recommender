# TODO: Watched-Movie Feedback Tuning

## Goal
Use already-watched movies as explicit taste signals for the recommender, especially for the “More or Less like this” evaluation flow.

## Tasks

### 1. Add richer feedback constants in `recommender.py`
- Define a shared feedback label map, for example:
  - `more_like_this`
  - `less_like_this`
  - `rewatchable`
  - `one_and_done`
  - `interesting_but_not_more`
  - `high_quality_not_my_taste`
  - `guilty_pleasure`
- Keep the existing `more_like_this` / `less_like_this` labels compatible with current saved feedback.

### 2. Update the feedback weight map in `add_feedback_similarity`
- Replace the current hardcoded two-label weights with the richer feedback constants.
- Make explicit watched-movie feedback stronger than passive ratings/likes.
- Preserve negative weighting for “less like this” style labels.
- Ensure direct feedback on a recommendation still adjusts that item directly.

### 3. Add a “Tune watched movies” UI block in `app.py`
- Add a section for watched/rated movies, likely on the Analysis page or sidebar.
- Show movie title, year, rating, metadata summary, and current feedback labels if present.
- Add buttons/selectbox for richer labels.
- Save selections through the existing feedback flow or `save_feedback_to_db`.
- Let users search/filter watched movies so the section does not become overwhelming.

### 4. Optional later: add `feedback_scope` or `note`
- Consider adding a `feedback_scope` column to distinguish:
  - recommendation feedback
  - watched-movie tuning feedback
  - evaluation feedback
- Consider adding a free-text `note` column for taste explanations.
- Not required for the initial implementation; the current generic `feedback` table can support the first version.

## Acceptance Criteria
- Watched movies can be marked with richer taste labels.
- Recommendation scoring changes based on those labels.
- Existing `more_like_this` and `less_like_this` feedback still works.
- The app no longer treats all high ratings as automatically meaning “more like this.”