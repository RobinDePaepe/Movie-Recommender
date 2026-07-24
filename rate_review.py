"""Reflection panel page: rate a film on fixed categories, get a suggested overall rating.

Evidence (director/cast/theme history) is kept as reference context, collapsed by default.
The main flow is category sliders + notes -> a computed suggestion -> an editable final rating
that can also be saved as the film's official rating.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st
from streamlit_extras.star_rating import star_rating
from streamlit_extras.stylable_container import stylable_container

from movie_database import import_tmdb_cache, load_latest_reflection, save_reflection
from reflection import REFLECTION_CATEGORIES, build_facets, resolve_for_reflection, suggested_overall_rating


def _history_table(history) -> pd.DataFrame:
    return pd.DataFrame(history, columns=["Title", "Year", "Your rating"])


def _slug(text: str) -> str:
    return text.lower().replace(" & ", "_").replace(" ", "_")


def _star_input(category: str, default_rating: float, mid: str) -> float:
    """10-star row (each star = half a point) for direct half-star click precision, no separate toggle."""
    star_key = f"reflect_star_{_slug(category)}_{mid}"
    if star_key not in st.session_state:
        st.session_state[star_key] = float(default_rating)
    current = st.session_state[star_key]

    with stylable_container(
        key=f"star_row_{_slug(category)}_{mid}",
        css_styles="""
        button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            min-height: unset !important;
            font-size: 1.35rem !important;
            line-height: 1 !important;
            color: #E6CF7A !important;
            transition: transform 0.1s ease !important;
        }
        button:hover {
            background: transparent !important;
            color: #F5E6A8 !important;
            transform: scale(1.2);
        }
        """,
    ):
        cols = st.columns(10, gap="small")
        for i, col in enumerate(cols):
            value = (i + 1) * 0.5
            with col:
                if st.button("★" if value <= current else "☆", key=f"{star_key}_btn_{i}"):
                    st.session_state[star_key] = value
                    st.rerun()

    rating_value = st.session_state[star_key]
    star_rating(rating_value)
    st.caption(f"{rating_value:.1f} / 5.0")
    return rating_value


def render_reflection_panel(
    data: Dict[str, pd.DataFrame],
    metadata: pd.DataFrame | None,
    tmdb_client=None,
    db_path=None,
    use_database: bool = False,
    cache_path=None,
) -> None:
    st.subheader("Reflection panel")
    st.caption(
        "Rate one film on the categories that matter to you, jot down why, and get a suggested "
        "overall rating built from those sub-ratings."
    )

    query = st.text_input(
        "Movie title or movie_id",
        placeholder="e.g. paris, texas (1984)",
        key="reflection_query",
    )
    if not query:
        return

    try:
        resolved = resolve_for_reflection(query, data, metadata, tmdb_client=tmdb_client)
    except ValueError as exc:
        st.error(str(exc))
        return

    mid = resolved["movie_id"]

    header_left, header_right = st.columns([1, 3])
    with header_left:
        if resolved.get("poster_url"):
            st.image(resolved["poster_url"], use_container_width=True)
    with header_right:
        st.markdown(f"### {resolved['title']} ({resolved['year']})")
        if resolved.get("genres"):
            st.caption(", ".join(resolved["genres"]))
        if resolved["in_db"] and resolved["rating"] is not None:
            st.info(f"Existing rating on file: {resolved['rating']:.1f}★ — saving below will overwrite it.")
        elif not resolved["in_db"]:
            st.caption("Not in your library — pulled fresh from TMDb.")

    with st.expander("Evidence from your history (director / cast / theme)"):
        facets = build_facets(resolved, data, metadata)
        facet_cols = st.columns(3)
        for col, facet in zip(facet_cols, facets):
            with col:
                label = facet["facet"].capitalize()
                value = f" — {facet['value']}" if facet.get("value") else ""
                st.markdown(f"**{label}**{value}")
                if facet["n"] == 0:
                    st.caption("No prior data.")
                else:
                    st.dataframe(_history_table(facet["history"]), hide_index=True, use_container_width=True)
                    st.caption(f"{facet['n']} prior film(s)")

    prior = load_latest_reflection(mid, db_path=db_path) if use_database and db_path else {"categories": {}, "overall": None}
    prior_categories = prior.get("categories", {})

    st.divider()
    st.markdown("#### Category ratings")
    category_ratings: Dict[str, float] = {}
    category_notes: Dict[str, str] = {}
    cols = st.columns(2)
    for i, category in enumerate(REFLECTION_CATEGORIES):
        default_rating = prior_categories.get(category, {}).get("rating", 3.0)
        default_note = prior_categories.get(category, {}).get("note", "")
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{category}**")
                category_ratings[category] = _star_input(category, float(default_rating), mid)
                category_notes[category] = st.text_area(
                    "Why?", value=default_note, key=f"reflect_note_{_slug(category)}_{mid}",
                    height=68, placeholder="What drove this rating?",
                )

    with st.expander("Category weights"):
        st.caption("Drag to change how much each category pulls the suggested overall rating.")
        weights: Dict[str, float] = {}
        for category in REFLECTION_CATEGORIES:
            weights[category] = st.slider(
                category, 0.0, 3.0, 1.0, 0.25, key=f"reflect_weight_{_slug(category)}_{mid}",
            )

    suggestion = suggested_overall_rating(category_ratings, weights)
    st.divider()
    st.markdown("#### Overall rating")
    st.caption(f"Weighted average of your category ratings: **{suggestion:.2f}**")

    prior_overall = prior.get("overall") or {}
    default_overall = prior_overall.get("rating", round(suggestion * 2) / 2)
    overall_rating = st.number_input(
        "Your rating", min_value=0.5, max_value=5.0, step=0.5,
        value=float(default_overall), key=f"reflect_overall_{mid}",
    )
    overall_note = st.text_area(
        "Overall notes", value=prior_overall.get("note", ""), key=f"reflect_overall_note_{mid}",
        placeholder="Anything that doesn't fit a single category — write it here.",
    )

    if not use_database or not db_path:
        st.caption("Saving reflections requires the SQLite database — build it from the sidebar first.")
        return

    if st.button("Save reflection", type="primary"):
        category_payload = {cat: (category_ratings[cat], category_notes[cat]) for cat in REFLECTION_CATEGORIES}
        save_reflection(
            mid, resolved["title"], resolved["year"], category_payload,
            overall_rating, overall_note, db_path=db_path,
        )
        if not resolved["in_db"] and cache_path is not None:
            import_tmdb_cache(cache_path=cache_path, db_path=db_path)
        st.success(f"Saved reflection for {resolved['title']} — official rating set to {overall_rating:.1f}★.")
        st.rerun()
