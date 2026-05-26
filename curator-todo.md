# TODO: Movie Curation Feature

## Goal
Add a “Curated Weeks” feature that builds an ordered watchlist around one anchor movie.

## 1. Add curator module
- Add `curator.py` to the project root.
- Use `build_curated_list()` to generate curated movie lists.
- Support custom list length with `total_movies`, defaulting to 7.

## 2. Add Streamlit page
- In `app.py`, add `"Curated Weeks"` to the sidebar page selector.
- Import:

```python
from curator import build_curated_list, anchor_options, CURATION_STYLES

##3. Add curator controls
Add anchor movie selectbox.
Add number slider: