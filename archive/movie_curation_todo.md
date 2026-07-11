# TODO: Movie Curation Feature

## Goal
Add a “Curated Weeks” feature that builds an ordered watchlist around one anchor movie.

---

## 1. Add curator module
- Add `curator.py` to the project root.
- Use `build_curated_list()` to generate curated movie lists.
- Support custom list length with `total_movies`, defaulting to 7.

### Core functionality
- Accept:
  - anchor movie
  - metadata
  - Letterboxd data
  - style
  - total movie count
- Generate ordered movie sequences.
- Assign roles to movies:
  - context
  - thematic setup
  - anchor
  - director connection
  - intensifier
  - contrast
  - afterglow

---

## 2. Add Streamlit page
- In `app.py`, add `"Curated Weeks"` to the sidebar page selector.

### Import curator functions
```python
from curator import (
    build_curated_list,
    anchor_options,
    CURATION_STYLES,
)
```

### Add page block
```python
elif page == "Curated Weeks":
    st.subheader("Curated movie week")
```

---

## 3. Add curator controls

### Anchor movie selector
- Select from:
  - watched movies
  - rated movies
  - watchlist movies

### Movie count slider
```python
total_movies = st.slider(
    "Number of movies",
    3,
    14,
    7
)
```

### Style selector
Use:
```python
CURATION_STYLES
```

Initial styles:
- Balanced
- Director-focused
- Theme-focused
- Vibe-focused
- Cinephile / historical context
- Gentler pacing

### Add checkboxes
- Allow watched movies
- Allow watchlisted movies
- Include anchor movie in final list

---

## 4. Render curated result

### Display format
Each item should show:
- Day / order number
- Role
- Movie title
- Year
- Why it was chosen
- Poster
- Genres / moods

### Nice-to-have
- Timeline / flow visualization
- “Intensity curve” across the week
- Group by emotional pacing

---

## 5. Add export functionality

### CSV export
Add:
```python
st.download_button(...)
```

### Future export ideas
- Markdown export
- Shareable JSON
- Letterboxd import format
- Printable “movie week card”

---

## 6. Optional database support

### Add tables
```sql
CREATE TABLE curated_weeks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor_movie_id TEXT,
    title TEXT,
    style TEXT,
    total_movies INTEGER,
    created_at TEXT
);

CREATE TABLE curated_week_items (
    week_id INTEGER,
    day_number INTEGER,
    movie_id TEXT,
    role TEXT,
    reason TEXT
);
```

### Future capabilities
- Save curated weeks
- Re-open previous weeks
- Favorite curated weeks
- Compare styles
- Track completion progress

---

## 7. Future improvements

### Recommendation quality
- Avoid emotionally heavy movies back-to-back
- Add pacing intelligence
- Detect duplicate directors / actors overload
- Improve contrast selection

### User interaction
- Regenerate one slot
- Pin movies manually
- Remove movie and refill automatically
- Drag-and-drop ordering

### Advanced curation modes
- “Film school”
- “Double feature”
- “Late-night”
- “Comfort week”
- “Criterion-core”
- “Actor deep dive”
- “Director evolution”

### LLM enhancements
- Generate natural-language explanations
- Create intro paragraph for the week
- Generate “why this order works”
- Generate discussion questions

### Visualization ideas
- Mood graph
- Decade distribution
- Genre balance
- Director network graph
- Influence chain visualization

### Long-term ideas
- Public curated week sharing
- Friend collaboration)
- Voting on curation variants
- AI-generated festival programming
- Seasonal curation events
- “Movie club mode”
