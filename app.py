from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging
import math
import os

# Streamlit's hot-reload watcher walks every imported module; when optional theme
# embeddings pull in `transformers`, its lazy vision submodules fail to import
# `torchvision` (which we don't use) and the watcher logs a traceback for each.
# Silence that one noisy logger without disabling hot-reload.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import pandas as pd
import plotly.express as px
import streamlit as st
import re
from streamlit_option_menu import option_menu
from streamlit_extras.stylable_container import stylable_container
from st_aggrid import AgGrid, GridOptionsBuilder

try:
    from dotenv import load_dotenv, set_key, find_dotenv
    load_dotenv()
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

from curator import CURATION_STYLES, anchor_options, build_curated_list
import llm_curator
import llm_providers
from rate_review import render_reflection_panel
from recommender import (
    ANCHOR_FOCUS_SCALE,
    apply_filters,
    available_filter_values,
    build_recommendations,
    ensure_export_dir,
    evaluate_historical_predictions,
    FEEDBACK_LABELS,
    TASTE_MODES,
    load_feedback,
    remove_feedback_from_csv,
    load_letterboxd,
    prepare_metadata,
    save_feedback,
)
from tmdb_client import TMDbClient, discover_movies_from_favorites, enrich_movies, metadata_from_cache
from letterboxd_sync import apply_sync_overlays, sync_rss, sync_status
from movie_database import (
    DB_PATH,
    apply_rss_overlays_to_db,
    database_status,
    import_feedback_csv,
    import_letterboxd_export,
    import_tmdb_cache,
    load_curated_week,
    load_curated_weeks,
    load_data_from_db,
    load_feedback_from_db,
    load_metadata_from_db,
    rebuild_database,
    save_curated_week,
    remove_feedback_from_db,
    save_feedback_to_db,
)

st.set_page_config(page_title="Personal Movie Recommender", layout="wide", page_icon="🎬")


def inject_theme() -> None:
    """Inject the premium dark-cinema styling once per session."""
    st.markdown(
        """
        <style>
        /* ---- Typography ---- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* ---- App background: subtle radial vignette ---- */
        .stApp {
            background:
                radial-gradient(1200px 600px at 50% -10%, #1c1c22 0%, #0E0E10 55%) fixed;
        }

        /* ---- Hero header ---- */
        .hero {
            padding: 1.6rem 0 1.2rem 0;
            border-bottom: 1px solid rgba(201,162,39,0.18);
            margin-bottom: 1.4rem;
        }
        .hero-title {
            font-family: 'Playfair Display', serif;
            font-size: 2.5rem;
            font-weight: 700;
            line-height: 1.05;
            margin: 0;
            background: linear-gradient(90deg, #F5E6A8 0%, #C9A227 60%, #9C7A12 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-sub {
            color: #9A9AA2;
            font-size: 0.95rem;
            letter-spacing: 0.02em;
            margin-top: 0.35rem;
        }
        .hero-mark {
            color: #C9A227;
            font-weight: 600;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.72rem;
        }

        /* ---- Score badge ---- */
        .score-badge {
            display: inline-flex;
            align-items: baseline;
            gap: 0.35rem;
            background: linear-gradient(135deg, rgba(201,162,39,0.18), rgba(201,162,39,0.06));
            border: 1px solid rgba(201,162,39,0.45);
            border-radius: 999px;
            padding: 0.35rem 0.95rem;
            margin: 0.4rem 0;
        }
        .score-badge .num {
            font-size: 1.35rem;
            font-weight: 700;
            color: #F5E6A8;
        }
        .score-badge .lbl {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #9A9AA2;
        }

        /* ---- Metadata chips ---- */
        .chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0; }
        .chip {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 6px;
            padding: 0.18rem 0.55rem;
            font-size: 0.74rem;
            color: #CFCFD6;
            white-space: nowrap;
        }
        .chip.accent { border-color: rgba(201,162,39,0.4); color: #E6CF7A; }

        /* ---- Poster images: rounded with depth + hover lift ---- */
        [data-testid="stImage"] img {
            border-radius: 10px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.55);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        [data-testid="stImage"] img:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(0,0,0,0.7), 0 0 0 1px rgba(201,162,39,0.35);
        }

        /* ---- Buttons: raised, clickable-looking controls instead of flat ghost boxes ---- */
        [data-testid="stBaseButton-secondary"] {
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.16);
            background: linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.02));
            color: #E8E6DE;
            font-weight: 600;
            padding: 0.5rem 1.15rem;
            box-shadow: 0 1px 0 rgba(255,255,255,0.07) inset, 0 2px 6px rgba(0,0,0,0.4);
            transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease, background 0.12s ease, color 0.12s ease;
        }
        [data-testid="stBaseButton-secondary"]:hover {
            border-color: rgba(201,162,39,0.6);
            color: #F5E6A8;
            background: linear-gradient(180deg, rgba(201,162,39,0.20), rgba(201,162,39,0.07));
            box-shadow: 0 1px 0 rgba(255,255,255,0.08) inset, 0 8px 18px rgba(0,0,0,0.5);
            transform: translateY(-1px);
        }
        [data-testid="stBaseButton-secondary"]:active {
            transform: translateY(0);
            box-shadow: 0 1px 3px rgba(0,0,0,0.5) inset;
        }
        [data-testid="stBaseButton-secondary"]:focus-visible {
            outline: 2px solid #C9A227;
            outline-offset: 2px;
        }
        [data-testid="stBaseButton-secondary"]:disabled {
            opacity: 0.4;
            transform: none;
            box-shadow: none;
        }

        /* Primary buttons (type="primary") get a solid gold fill so the one key action per view stands out */
        [data-testid="stBaseButton-primary"] {
            border: 1px solid rgba(201,162,39,0.75);
            background: linear-gradient(135deg, #F5E6A8 0%, #C9A227 65%, #9C7A12 100%);
            color: #1C1408;
            font-weight: 700;
            padding: 0.5rem 1.15rem;
            box-shadow: 0 4px 14px rgba(201,162,39,0.35);
            transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
        }
        [data-testid="stBaseButton-primary"]:hover {
            background: linear-gradient(135deg, #F8ECBC 0%, #D8AF3A 65%, #A9860F 100%);
            border-color: rgba(201,162,39,0.95);
            box-shadow: 0 8px 22px rgba(201,162,39,0.5);
            transform: translateY(-1px);
        }
        [data-testid="stBaseButton-primary"]:active {
            transform: translateY(0);
            box-shadow: 0 2px 8px rgba(201,162,39,0.4);
        }
        [data-testid="stBaseButton-primary"]:focus-visible {
            outline: 2px solid #F5E6A8;
            outline-offset: 2px;
        }

        @media (prefers-reduced-motion: reduce) {
            [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
                transition: none;
            }
        }

        /* ---- Metric cards ---- */
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
        }

        /* ---- Expanders ---- */
        [data-testid="stExpander"] {
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 10px;
            background: rgba(255,255,255,0.02);
        }

        /* ---- Headings ---- */
        h1, h2, h3 { letter-spacing: -0.01em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-mark">🎬 Your Personal Cinema</div>
            <h1 class="hero-title">Movie Recommender</h1>
            <div class="hero-sub">Letterboxd + TMDb · taste-aware picks, curated weeks, and discovery tuned to you.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_badge_html(score: float) -> str:
    return (
        f'<div class="score-badge"><span class="num">{score:.2f}</span>'
        f'<span class="lbl">match score</span></div>'
    )


def chips_html(items: list, accent: bool = False) -> str:
    cls = "chip accent" if accent else "chip"
    spans = "".join(f'<span class="{cls}">{str(x)}</span>' for x in items if str(x).strip())
    return f'<div class="chips">{spans}</div>' if spans else ""


inject_theme()
render_hero()


def fmt_year(value) -> str:
    """Render a Year value without the trailing '.0' pandas float coercion leaves behind."""
    if pd.isna(value) or str(value).strip() in ("", "nan", "<NA>"):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def render_reasons(text: str, sep: str = ";") -> None:
    """Render a separator-delimited reason string as markdown bullet points.

    sep: delimiter string, commonly ';' for reasons or ',' for lists.
    """
    if not text:
        return
    s = str(text)
    if sep == ";":
        parts = [p.strip() for p in re.split(r';\s*', s) if p.strip()]
    elif sep == ",":
        parts = [p.strip() for p in s.split(',') if p.strip()]
    else:
        parts = [p.strip() for p in s.split(sep) if p.strip()]
    if not parts:
        return
    md = "\n".join(f"- {p}" for p in parts)
    st.markdown(md)


def render_grid(df: pd.DataFrame, height: int = 420) -> None:
    """Sortable/filterable data grid (AgGrid) with list-type columns flattened to text."""
    grid_df = df.copy()
    for col in grid_df.columns:
        if grid_df[col].apply(lambda v: isinstance(v, list)).any():
            grid_df[col] = grid_df[col].apply(lambda v: ", ".join(map(str, v)) if isinstance(v, list) else v)

    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_default_column(sortable=True, filter=True, resizable=True, wrapText=True, autoHeight=False)
    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=20)
    AgGrid(
        grid_df,
        gridOptions=gb.build(),
        theme="streamlit",
        height=height,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=False,
    )


export_zip = Path("data/letterboxd_export.zip")
if not export_zip.exists():
    st.error("Put your Letterboxd export zip at data/letterboxd_export.zip")
    st.stop()

db_path = Path("data/movie_recommender.sqlite")
use_database = db_path.exists()

# --- Startup auto-sync ---
# Runs once per session if LETTERBOXD_USERNAME (or LETTERBOXD_RSS_CLIENT_ID, a direct RSS
# URL) is set and last sync was > 1 hour ago. Updates overlay CSVs (and SQLite if active)
# before data loads, so the session starts fresh.
if "auto_synced_this_session" not in st.session_state:
    st.session_state.auto_synced_this_session = False

_lb_auto_user = os.getenv("LETTERBOXD_USERNAME", "") or os.getenv("LETTERBOXD_RSS_CLIENT_ID", "")
_auto_new_events = 0
if not st.session_state.auto_synced_this_session and _lb_auto_user:
    _sync_state = sync_status()
    _last_sync = _sync_state.get("last_sync_at", "")
    _needs_sync = True
    if _last_sync:
        try:
            _last_dt = datetime.fromisoformat(_last_sync)
            if _last_dt.tzinfo is None:
                _last_dt = _last_dt.replace(tzinfo=timezone.utc)
            _needs_sync = (datetime.now(timezone.utc) - _last_dt) > timedelta(hours=1)
        except Exception:
            pass
    if _needs_sync:
        try:
            _auto_result = sync_rss(_lb_auto_user)
            _auto_new_events = _auto_result.get("new_events", 0)
            if use_database:
                apply_rss_overlays_to_db(db_path=db_path)
        except Exception:
            pass
    st.session_state.auto_synced_this_session = True

if use_database:
    data = load_data_from_db(db_path)
else:
    export_dir = ensure_export_dir(export_zip)
    base_data = load_letterboxd(export_dir)
    data = apply_sync_overlays(base_data)

_movie_cols = ["Name", "Year", "movie_id"]
movie_frames = [
    data["ratings"].reindex(columns=_movie_cols),
    data["watched"].reindex(columns=_movie_cols),
    data["watchlist"].reindex(columns=_movie_cols),
    data["likes"].reindex(columns=_movie_cols),
]
if not data["lists"].empty:
    movie_frames.append(data["lists"].reindex(columns=_movie_cols))
all_movies = pd.concat(movie_frames, ignore_index=True).drop_duplicates()

if _auto_new_events > 0:
    st.toast(f"Auto-synced Letterboxd: {_auto_new_events} new events added.")

def get_tmdb_api_key() -> str:
    """TMDb key entered on the Data & Sync page, falling back to the environment."""
    return st.session_state.get("tmdb_api_key_input") or os.getenv("TMDB_API_KEY", "")


cache_path = Path("data/tmdb_cache.json")
if use_database:
    metadata = load_metadata_from_db(db_path)
    # Coverage should reflect metadata found for movies you actually track (all_movies),
    # not every row in movie_metadata (which also includes discovered-but-untracked candidates).
    known_ids = set(all_movies.get("movie_id", pd.Series(dtype=str)).dropna())
    metadata_known = metadata[metadata.get("movie_id", pd.Series(dtype=str)).isin(known_ids)] if not metadata.empty else metadata
    feedback = load_feedback_from_db(db_path)
else:
    metadata = metadata_from_cache(None, cache_path=cache_path, include_all=True)
    metadata_known = metadata_from_cache(all_movies, cache_path=cache_path)
    feedback = load_feedback()

cached_count = len(metadata) if not metadata.empty else 0
known_count = len(metadata_known) if not metadata_known.empty else 0
found_count = int(metadata.get("tmdb_found", pd.Series(dtype=bool)).fillna(False).sum()) if not metadata.empty else 0

ALL_MOODS = ["Tense", "Emotional", "Gritty", "Exciting", "Imaginative", "Light", "Reflective"]

PAGES = ["Tonight's Pick", "Recommendations", "Curated Weeks", "Analysis", "Evaluation", "Reflection", "Data & Sync"]
PAGE_ICONS = ["moon-stars", "bullseye", "calendar-week", "bar-chart", "clipboard-data", "chat-heart", "gear"]

with st.sidebar:
    page = option_menu(
        menu_title=None,
        options=PAGES,
        icons=PAGE_ICONS,
        default_index=0,
        key="nav_menu",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#9A9AA2", "font-size": "15px"},
            "nav-link": {
                "font-size": "14px",
                "font-weight": "500",
                "text-align": "left",
                "margin": "2px 0",
                "padding": "10px 14px",
                "border-radius": "8px",
                "color": "#CFCFD6",
                "--hover-color": "rgba(201,162,39,0.10)",
            },
            "nav-link-selected": {
                "background-color": "rgba(201,162,39,0.16)",
                "color": "#F5E6A8",
                "font-weight": "700",
            },
        },
    )

st.sidebar.divider()
st.sidebar.caption("THIS SESSION")
mode_label = st.sidebar.radio(
    "Recommendation source",
    ["My watchlist", "Not on my watchlist"],
    help="Outside-watchlist recommendations use your lists plus TMDb-discovered cached records, excluding watched/rated/watchlisted movies.",
)
mode = "outside_watchlist" if mode_label == "Not on my watchlist" else "watchlist"

filter_values_preview = available_filter_values(pd.DataFrame())
taste_mode = st.sidebar.selectbox("Taste mode", filter_values_preview.get("taste_modes", ["Balanced"]), index=0)

with st.sidebar.expander("Advanced scoring weights"):
    st.caption("Drag to change how much each signal pulls the final score.")
    content_weight = st.slider("Taste similarity", 0.0, 3.0, 1.0, 0.25, help="How strongly TF-IDF content similarity to your high-rated films affects the score.")
    theme_weight = st.slider("Theme similarity", 0.0, 3.0, 1.0, 0.25, help="How strongly conceptual/thematic similarity (what a film is *about* — keywords + overview) to your high-rated films affects the score. Independent of genre/director/cast.")
    entity_weight = st.slider("Director / cast influence", 0.0, 3.0, 1.0, 0.25, help="How strongly a shared director, writer, or cast member you've rated highly affects the score.")
    list_weight = st.slider("List signals", 0.0, 3.0, 1.0, 0.25, help="How much being on your curated lists counts.")
    anchor_weight = st.slider("Anchor influence", 0.0, 3.0, 1.0, 0.25, help="How strongly the film you anchor on (Recommendations page) pulls thematically similar candidates up.")
    feedback_weight = st.slider("Watched-movie feedback", 0.0, 3.0, 1.0, 0.25, help="How strongly the taste labels you give watched films (Analysis → Tune watched movies) pull recommendations toward or away from similar films. Deliberate tuning already counts more than passive feedback.")
score_weights = {"content": content_weight, "theme": theme_weight, "entity": entity_weight, "list": list_weight, "anchor": anchor_weight, "feedback": feedback_weight}

st.sidebar.divider()
_sync_brief = sync_status()
st.sidebar.caption(
    f"🔄 {cached_count} cached · {'SQLite' if use_database else 'CSV/JSON'} backend · "
    f"Last sync: {_sync_brief.get('last_sync_at', 'never')}"
)

recs, decade_prefs = build_recommendations(data, metadata=metadata, mode=mode, feedback=feedback, taste_mode=taste_mode, score_weights=score_weights)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Rated", len(data["ratings"]))
col2.metric("Watched", len(data["watched"]))
col3.metric("Watchlist", len(data["watchlist"]))
col4.metric("Custom list entries", len(data["lists"]))
col5.metric("Metadata coverage", f"{known_count}/{len(all_movies)}")


def store_feedback(movie_id: str, feedback_value: str, scope: str = "recommendation") -> None:
    if use_database:
        save_feedback_to_db(movie_id, feedback_value, scope=scope, db_path=db_path)
    else:
        save_feedback(movie_id, feedback_value, scope=scope)


def remove_feedback(movie_id: str, labels: list) -> None:
    if use_database:
        remove_feedback_from_db(movie_id, labels, db_path=db_path)
    else:
        remove_feedback_from_csv(movie_id, labels)


def poster_card(row: pd.Series, idx: int) -> None:
    with stylable_container(
        key=f"poster_card_{idx}_{row.get('movie_id')}",
        css_styles="""
        {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 14px 14px 10px 14px;
            background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012));
            box-shadow: 0 6px 20px rgba(0,0,0,0.35);
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;

            &:hover {
                transform: translateY(-3px);
                border-color: rgba(201,162,39,0.35);
                box-shadow: 0 14px 32px rgba(0,0,0,0.5);
            }
        }
        """,
    ):
        title = f"{row.get('Name', '')} ({fmt_year(row.get('Year'))})"
        _pu = row.get("poster_url")
        if pd.notna(_pu) and str(_pu).strip():
            st.image(str(_pu).strip(), use_container_width=True)
        else:
            st.info("No poster")
        st.markdown(f"**{title}**")
        directors = row.get("directors", [])
        if isinstance(directors, list) and directors:
            st.caption(f"Dir. {', '.join(directors[:2])}")
        st.markdown(score_badge_html(float(row.get("score", 0) or 0)), unsafe_allow_html=True)
        rt = row.get("runtime", "")
        chip_items = [f"{rt} min"] if rt else []
        genres = row.get("genres", [])
        if isinstance(genres, list):
            chip_items.extend(genres[:3])
        mood_items = row.get("moods", [])
        mood_items = mood_items[:2] if isinstance(mood_items, list) else []
        if chip_items:
            st.markdown(chips_html(chip_items), unsafe_allow_html=True)
        if mood_items:
            st.markdown(chips_html(mood_items, accent=True), unsafe_allow_html=True)
        if row.get("why"):
            st.caption(str(row.get("why")))
        _disc = row.get("discovered_from")
        if pd.notna(_disc) and str(_disc).strip():
            st.caption(f"🔎 Discovered from: {str(_disc).strip()}")
        with st.expander("Why?"):
            render_reasons(row.get("why_details", row.get("why", "")))
            if row.get("overview"):
                st.write(row.get("overview"))
        b1, b2 = st.columns(2)
        if b1.button("More", key=f"more_{idx}_{row.get('movie_id')}"):
            store_feedback(row["movie_id"], "more_like_this")
            st.rerun()
        if b2.button("Less", key=f"less_{idx}_{row.get('movie_id')}"):
            store_feedback(row["movie_id"], "less_like_this")
            st.rerun()


def curated_week_card(row: pd.Series) -> None:
    genres = row.get("genres", []) if isinstance(row.get("genres"), list) else []
    moods = row.get("moods", []) if isinstance(row.get("moods"), list) else []
    runtime = row.get("runtime")

    left, right = st.columns([1, 4])
    with left:
        _pu = row.get("poster_url")
        if pd.notna(_pu) and str(_pu).strip():
            st.image(str(_pu).strip(), use_container_width=True)
        else:
            st.info("No poster")
    with right:
        st.markdown(f"### Day {int(row.get('day', 0))}: {row.get('Name', '')} ({fmt_year(row.get('Year'))})")
        directors = row.get("directors", [])
        director_caption = f"Dir. {', '.join(directors[:2])} | " if isinstance(directors, list) and directors else ""
        st.caption(f"{director_caption}{row.get('role', '')} | {row.get('role_description', '')}")
        st.write(row.get("why", ""))
        if genres:
            st.caption("Genres: " + ", ".join(genres[:4]))
        if moods:
            st.caption("Moods: " + ", ".join(moods[:4]))
        if pd.notna(runtime) and str(runtime).strip():
            st.caption(f"Runtime: {runtime} min")
        if row.get("overview"):
            with st.expander("Overview", expanded=False):
                st.write(row.get("overview"))
        tmdb_url = row.get("tmdb_url")
        if isinstance(tmdb_url, str) and tmdb_url:
            st.link_button("Open in TMDb", tmdb_url)


def enrich_llm_pick(pick: dict, client: "TMDbClient | None") -> dict:
    """Look up poster/overview/runtime for an LLM-suggested title via TMDb."""
    if client is None:
        return {}
    try:
        year = pick.get("year")
        year = int(year) if year not in (None, "", 0) else None
        rec = client.fetch_movie_metadata(pick.get("title", ""), year)
        return rec if isinstance(rec, dict) and rec.get("tmdb_found") else {}
    except Exception:
        return {}


def llm_pick_card(pick: dict, meta: dict, digest: "llm_curator.TasteDigest | None") -> None:
    cat = llm_curator.CATEGORIES.get(pick.get("category"), {"label": pick.get("category", ""), "accent": "#E8B04B"})
    title = pick.get("title", "")
    director = pick.get("director") or (", ".join(meta.get("directors", [])[:2]) if meta.get("directors") else "")
    year = pick.get("year") or meta.get("year") or ""
    poster = meta.get("poster_url")
    is_seen = bool(digest and digest.loaded and llm_curator._norm_title(title) in digest.seen_keys)

    left, right = st.columns([1, 4])
    with left:
        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.info("No poster")
    with right:
        st.caption(cat["label"].upper())
        seen_tag = "  ·  ✓ already seen" if (is_seen and pick.get("category") != "rewatch") else ""
        st.markdown(f"### {title} ({fmt_year(year)}){seen_tag}")
        if director:
            st.caption(f"Dir. {director}")
        if pick.get("reason"):
            st.write(pick.get("reason"))
        if meta.get("overview"):
            with st.expander("Overview", expanded=False):
                st.write(meta.get("overview"))
        runtime = meta.get("runtime")
        if runtime:
            st.caption(f"Runtime: {runtime} min")
        tmdb_url = meta.get("tmdb_url")
        if tmdb_url:
            st.link_button("Open in TMDb", tmdb_url)


def _component_explanations(row: pd.Series, taste_mode: str = "Balanced") -> dict:
    """Plain-language 'why this number' text for each score_breakdown bar.

    Mirrors the same underlying columns as explain_detailed(), but keyed by the
    exact component labels used in the chart and without a high activation
    threshold, so every bar that's actually drawn gets an explanation.
    """
    text: dict = {}

    decade = row.get("decade", "")
    avg_user_rating = row.get("avg_user_rating")
    decade_bits = []
    if pd.notna(avg_user_rating) and decade:
        decade_bits.append(f"you rate {decade} films {float(avg_user_rating):.1f}★ on average")
    if float(row.get("liked_decade_bonus", 0) or 0) > 0:
        decade_bits.append("shares a decade with films you liked")
    if float(row.get("recency_bonus", 0) or 0) > 0:
        decade_bits.append("recent release bonus")
    if decade_bits:
        text["Decade & recency"] = "; ".join(decade_bits).capitalize() + "."

    list_count = int(row.get("list_count", 0) or 0)
    if list_count > 0:
        names = row.get("list_names_full") or row.get("list_names") or ""
        text["List signals"] = f"In {list_count} list(s): {names}." if names else f"In {list_count} list(s)."

    taste_matches = row.get("taste_matches_full") or row.get("taste_matches") or ""
    if taste_matches:
        text["Taste similarity"] = f"Matches your high-rated films on {taste_matches}."
    elif float(row.get("content_score", 0) or 0) < 0:
        text["Taste similarity"] = "Similar to films you rated poorly."

    theme_score = float(row.get("theme_score", 0) or 0)
    if abs(theme_score) > 0.01:
        text["Theme similarity"] = (
            "Explores similar keywords/concepts to films you rate highly."
            if theme_score > 0
            else "Keywords/concepts diverge from films you rate highly."
        )

    feedback_score = float(row.get("feedback_score", 0) or 0)
    if abs(feedback_score) > 0.01:
        text["Feedback"] = (
            "Similar to movies you tagged as 'more like this' (or direct feedback on this film)."
            if feedback_score > 0
            else "Similar to movies you tagged as 'less like this' (or direct feedback on this film)."
        )

    if float(row.get("taste_mode_score", 0) or 0) > 0:
        text["Taste mode"] = f"Fits the selected taste mode: {taste_mode}."

    entity_score = float(row.get("entity_score", 0) or 0)
    if abs(entity_score) > 0.01:
        text["Dir / Cast affinity"] = (
            "Directed/written/starring someone you've consistently rated highly."
            if entity_score > 0
            else "Involves a director/writer/cast member you've rated poorly in the past."
        )

    if float(row.get("anchor_score", 0) or 0) > 0.01:
        text["Anchor match"] = "Thematically similar to your anchored film."

    if float(row.get("mood_penalty", 0) or 0) > 0:
        text["Mood penalty"] = "Matches a mood you're avoiding this session."

    return text


def render_score_breakdown(row: pd.Series, score_weights: dict, anchor_active: bool = False, taste_mode: str = "Balanced") -> None:
    content_w = float(score_weights.get("content", 1.0))
    theme_w = float(score_weights.get("theme", 1.0))
    entity_w = float(score_weights.get("entity", 1.0))
    list_w = float(score_weights.get("list", 1.0))
    anchor_w = float(score_weights.get("anchor", 1.0))
    # Mirror build_recommendations' anchor-focus attenuation so the chart stays honest.
    if anchor_active:
        content_w *= ANCHOR_FOCUS_SCALE
        theme_w *= ANCHOR_FOCUS_SCALE
        entity_w *= ANCHOR_FOCUS_SCALE

    list_contrib = float(row.get("list_contribution", 0) or 0)
    heuristic = float(row.get("heuristic_score", 3.0) or 3.0)
    base_delta = heuristic - list_contrib - 3.0  # decade + recency above the 3.0 baseline

    components = [
        ("Decade & recency", base_delta),
        ("List signals", list_contrib * list_w),
        ("Taste similarity", float(row.get("content_score", 0) or 0) * content_w),
        ("Theme similarity", float(row.get("theme_score", 0) or 0) * theme_w),
        ("Feedback", float(row.get("feedback_score", 0) or 0)),
        ("Taste mode", float(row.get("taste_mode_score", 0) or 0)),
        ("Dir / Cast affinity", float(row.get("entity_score", 0) or 0) * entity_w),
        ("Anchor match", float(row.get("anchor_score", 0) or 0) * anchor_w),
        ("Mood penalty", -float(row.get("mood_penalty", 0) or 0)),
    ]
    components = [(label, val) for label, val in components if abs(val) > 0.01]

    if not components:
        st.caption("No score contribution data available.")
        return

    df_breakdown = pd.DataFrame(components, columns=["Component", "Contribution"])
    df_breakdown["Direction"] = df_breakdown["Contribution"].apply(lambda v: "Positive" if v >= 0 else "Negative")
    df_breakdown = df_breakdown.sort_values("Contribution")

    fig = px.bar(
        df_breakdown,
        x="Contribution",
        y="Component",
        orientation="h",
        color="Direction",
        color_discrete_map={"Positive": "#4CAF50", "Negative": "#EF5350"},
    )
    fig.update_layout(
        height=max(200, len(components) * 38 + 80),
        margin=dict(l=0, r=20, t=10, b=20),
        xaxis_title="Contribution to score",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    explanations = _component_explanations(row, taste_mode)
    lines = [f"- **{label}** ({val:+.2f}): {explanations[label]}" for label, val in components if label in explanations]
    if lines:
        st.markdown("\n".join(lines))


if page == "Tonight's Pick":
    st.subheader("What should I watch tonight?")
    st.caption("One pick, committed. Adjust until it clicks — then hit 'This is perfect!'")

    tp1, tp2 = st.columns(2)
    with tp1:
        tonight_taste = st.selectbox(
            "I'm in the mood for...",
            list(TASTE_MODES.keys()),
            index=0,
            key="tonight_taste",
        )
    with tp2:
        _time_options = {"Any length": None, "Under 100 min": (0, 100), "Under 2 hours": (0, 120), "Under 3 hours": (0, 180)}
        tonight_time_label = st.segmented_control(
            "Time available", list(_time_options.keys()), default="Any length", key="tonight_time",
        )
        tonight_runtime = _time_options.get(tonight_time_label)

    with st.popover("Not in the mood for..."):
        tonight_avoid = st.segmented_control(
            "Avoid these moods", ALL_MOODS, selection_mode="multi", default=[], key="tonight_avoid_moods",
        ) or []

    if "tonight_skipped" not in st.session_state:
        st.session_state.tonight_skipped = []

    tonight_recs, _ = build_recommendations(
        data, metadata=metadata, mode=mode, feedback=feedback,
        taste_mode=tonight_taste,
        score_weights=score_weights,
        avoid_moods=tonight_avoid or [],
    )

    if tonight_runtime:
        tonight_recs = apply_filters(tonight_recs, runtime_range=tonight_runtime)

    tonight_recs = tonight_recs[~tonight_recs["movie_id"].isin(st.session_state.tonight_skipped)].reset_index(drop=True)

    if tonight_recs.empty:
        st.warning("No movies match your current filters. Try adjusting mood or runtime, or reset skipped movies.")
        if st.button("Reset skipped movies"):
            st.session_state.tonight_skipped = []
            st.rerun()
    else:
        pick = tonight_recs.iloc[0]

        pc1, pc2 = st.columns([1, 2])
        with pc1:
            _pu = pick.get("poster_url")
            if pd.notna(_pu) and str(_pu).strip():
                st.image(str(_pu).strip(), use_container_width=True)
            else:
                st.info("No poster")
        with pc2:
            st.markdown(f"## {pick.get('Name', '')} ({fmt_year(pick.get('Year'))})")
            pick_directors = pick.get("directors", [])
            if isinstance(pick_directors, list) and pick_directors:
                st.caption(f"Dir. {', '.join(pick_directors[:2])}")
            chip_items = []
            rt = pick.get("runtime")
            if rt and str(rt).strip() not in ("", "nan", "<NA>"):
                chip_items.append(f"{rt} min")
            genres = pick.get("genres", [])
            if isinstance(genres, list):
                chip_items.extend(genres[:3])
            mood_items = pick.get("moods", [])
            mood_items = mood_items[:3] if isinstance(mood_items, list) else []
            html = score_badge_html(float(pick.get("score", 0) or 0))
            if chip_items:
                html += chips_html(chip_items)
            if mood_items:
                html += chips_html(mood_items, accent=True)
            st.markdown(html, unsafe_allow_html=True)
            _disc = pick.get("discovered_from")
            if pd.notna(_disc) and str(_disc).strip():
                st.caption(f"🔎 Discovered from: {str(_disc).strip()}")
            with st.expander("Why this?", expanded=True):
                render_reasons(str(pick.get("why_details") or pick.get("why", "")))
                if pick.get("overview"):
                    st.write(str(pick["overview"]))

        with st.expander("Score breakdown", expanded=False):
            render_score_breakdown(pick, score_weights, taste_mode=tonight_taste)

        st.divider()
        ba, bb, bc = st.columns(3)
        if ba.button("This is perfect!", use_container_width=True):
            store_feedback(pick["movie_id"], "more_like_this")
            st.session_state.tonight_skipped = []
            st.success(f"Enjoy **{pick.get('Name')}**! Saved as 'more like this'.")
        if bb.button("Give me another", use_container_width=True):
            st.session_state.tonight_skipped.append(pick["movie_id"])
            st.rerun()
        if bc.button("Not for me", use_container_width=True):
            store_feedback(pick["movie_id"], "less_like_this")
            st.session_state.tonight_skipped.append(pick["movie_id"])
            st.rerun()

        skipped_n = len(st.session_state.tonight_skipped)
        if skipped_n > 0:
            sk1, sk2 = st.columns([3, 1])
            sk1.caption(f"Skipped {skipped_n} movie(s) this session.")
            if sk2.button("Reset skipped"):
                st.session_state.tonight_skipped = []
                st.rerun()

elif page == "Recommendations":
    st.subheader("Recommended next watches" if mode == "watchlist" else "Recommended outside your watchlist")
    if metadata.empty:
        st.info("TMDb cache is empty. The app is using the original list/decade ranking until you fetch metadata.")
    else:
        st.success("Using TMDb metadata for content similarity, discovery candidates, mood filters, and feedback similarity.")

    # --- Anchor film ---
    anchor_movie_id = None
    anchor_focus = True
    with st.popover("Anchor on a film"):
        st.caption("Pick a film you love and the engine will boost candidates that are thematically similar to it.")
        if metadata.empty:
            st.info("Fetch TMDb metadata first to enable film anchoring.")
        else:
            from recommender import prepare_metadata as _prep_meta
            anchor_pool = _prep_meta(metadata)
            watched_ids = set(data["watched"].get("movie_id", pd.Series(dtype=str)).dropna())
            rated_ids = set(data["ratings"].get("movie_id", pd.Series(dtype=str)).dropna())
            anchor_pool = anchor_pool[anchor_pool["movie_id"].isin(watched_ids | rated_ids)].copy()
            anchor_pool = anchor_pool[anchor_pool["feature_text"].str.len().gt(0)].sort_values("Name")
            if anchor_pool.empty:
                st.info("No watched/rated films with metadata found.")
            else:
                anchor_labels = ["— none —"] + [f"{r.Name} ({r.Year})" for r in anchor_pool.itertuples()]
                anchor_choice = st.selectbox("Film to anchor on", anchor_labels)
                anchor_focus = st.checkbox(
                    "Anchor focus", value=True,
                    help="Let the anchored film lead by easing off your general taste profile, so concept-cousins aren't cancelled out.",
                )
                if anchor_choice != "— none —":
                    chosen_idx = anchor_labels.index(anchor_choice) - 1
                    anchor_movie_id = str(anchor_pool.iloc[chosen_idx]["movie_id"])
                    st.caption(f"Boosting candidates thematically similar to: **{anchor_choice}**")

    # --- Mood avoidance ---
    with st.popover("Not in the mood for..."):
        st.caption("Temporarily penalise these moods in this session. No permanent feedback saved.")
        avoid_moods = st.segmented_control("Avoid tonight", ALL_MOODS, selection_mode="multi", default=[], key="avoid_moods_rec") or []

    # Re-run scoring if anchor or mood avoidance is active
    if anchor_movie_id or avoid_moods:
        from recommender import build_recommendations as _build_recs
        recs, decade_prefs = _build_recs(
            data, metadata=metadata, mode=mode, feedback=feedback, taste_mode=taste_mode,
            score_weights=score_weights, anchor_movie_id=anchor_movie_id, avoid_moods=avoid_moods,
            anchor_focus=anchor_focus,
        )

    filter_values = available_filter_values(recs)
    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        selected_moods = f1.multiselect("Mood", filter_values.get("moods", []))
        selected_decades = f2.multiselect("Decade", filter_values.get("decades", []))
        selected_genres = f3.multiselect("Genre", filter_values.get("genres", []))
        f4, f5, f6 = st.columns(3)
        selected_languages = f4.multiselect("Language", filter_values.get("languages", []))
        runtime_values = pd.to_numeric(recs.get("runtime", pd.Series(dtype=float)), errors="coerce").dropna()
        if not runtime_values.empty:
            min_rt, max_rt = int(runtime_values.min()), int(runtime_values.max())
            runtime_range = f5.slider("Runtime", min_rt, max_rt, (min_rt, max_rt), help="Movies without runtime metadata are kept in results.")
        else:
            runtime_range = None
            f5.caption("Runtime filter appears after TMDb metadata is cached.")
        query = f6.text_input("Search title/list/metadata")

    filtered = apply_filters(recs, genres=selected_genres, languages=selected_languages, moods=selected_moods, decades=selected_decades, runtime_range=runtime_range, query=query)
    anchor_note = f" | Anchor: {anchor_choice}" if anchor_movie_id else ""
    mood_note = f" | Avoiding: {', '.join(avoid_moods)}" if avoid_moods else ""
    st.caption(f"Showing {min(100, len(filtered))} of {len(filtered)} recommendations. Taste mode: {taste_mode}{anchor_note}{mood_note}.")

    view = st.segmented_control("View", ["Poster cards", "Table"], default="Poster cards", key="rec_view") or "Poster cards"
    if view == "Poster cards":
        top = filtered.head(12).reset_index(drop=True)
        for start in range(0, len(top), 4):
            cols = st.columns(4)
            for offset, col in enumerate(cols):
                idx = start + offset
                if idx < len(top):
                    with col:
                        poster_card(top.iloc[idx], idx)
    else:
        show_cols = ["Name", "Year", "score", "heuristic_score", "content_score", "theme_score", "feedback_score", "taste_mode_score", "entity_score", "anchor_score", "mood_penalty", "why", "Letterboxd URI"]
        show_cols += [c for c in ["genres", "moods", "runtime", "languages", "directors", "cast", "keywords", "tmdb_url", "discovered_from"] if c in filtered.columns]
        render_grid(filtered[show_cols].head(100))

    details_frame = filtered.head(100)[["Name", "Year", "movie_id", "why", "why_details", "list_names_full", "taste_matches_full"]].copy()
    if not details_frame.empty:
        labels = [f"{r.Name} ({r.Year})" for r in details_frame.itertuples()]
        sel = st.selectbox("Show details for", ["- none -"] + labels)
        if sel and sel != "- none -":
            sel_idx = labels.index(sel)
            row = details_frame.iloc[sel_idx]
            with st.expander("Why this recommendation?", expanded=True):
                render_reasons(row["why_details"] or row["why"])
            with st.expander("Matched lists & taste matches", expanded=False):
                    lists = row.get("list_names_full", "")
                    tastes = row.get("taste_matches_full", "")
                    if lists:
                        st.write("Lists:")
                        render_reasons(lists, sep=",")
                    if tastes:
                        st.write("Taste matches:")
                        render_reasons(tastes)
            with st.expander("Score breakdown", expanded=False):
                full_row = filtered.head(100).iloc[sel_idx]
                render_score_breakdown(full_row, score_weights, anchor_active=bool(anchor_movie_id and anchor_focus), taste_mode=taste_mode)

    st.download_button("Download recommendations as CSV", filtered.to_csv(index=False).encode("utf-8"), "movie_recommendations.csv", "text/csv")

    st.subheader("Feedback")
    if not feedback.empty:
        st.caption(f"Stored feedback events: {len(feedback)}. Feedback now affects movies similar to the selected film, not only the selected title.")
    else:
        st.caption("Use More/Less on poster cards or below to tune the model.")
    feedback_options = filtered.head(25)[["Name", "Year", "movie_id"]].copy()
    if not feedback_options.empty:
        labels = [f"{r.Name} ({r.Year})" for r in feedback_options.itertuples()]
        selected_label = st.selectbox("Choose a recommendation to tune", labels)
        selected_row = feedback_options.iloc[labels.index(selected_label)]
        b1, b2 = st.columns(2)
        if b1.button("More like this"):
            store_feedback(selected_row["movie_id"], "more_like_this")
            st.rerun()
        if b2.button("Less like this"):
            store_feedback(selected_row["movie_id"], "less_like_this")
            st.rerun()

    st.subheader("Your rating affinity by decade")
    if not decade_prefs.empty:
        fig = px.bar(decade_prefs, x="decade", y="avg_user_rating", hover_data=["decade_score"])
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("How scoring works"):
        st.write(
            "The recommender combines Letterboxd heuristics, TMDb content similarity, similarity-based feedback, and the selected taste mode. "
            "Outside-watchlist discovery can add new TMDb candidates from movies similar to your high-rated films."
        )

elif page == "Analysis":
    st.subheader("Personal movie analysis")
    ratings_df = data["ratings"].copy()
    diary_df = data["diary"].copy()
    watched_df = data["watched"].copy()
    if ratings_df.empty and watched_df.empty:
        st.info("No watched/rating data loaded yet.")
    else:
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Watched movies", len(watched_df))
        a2.metric("Rated movies", len(ratings_df))
        a3.metric("Diary / watch events", len(diary_df))
        if not diary_df.empty and "Rewatch" in diary_df.columns:
            a4.metric("Rewatch events", int(pd.to_numeric(diary_df["Rewatch"], errors="coerce").fillna(0).sum()))
        else:
            a4.metric("Rewatch events", 0)

        if not ratings_df.empty:
            ratings_df["Rating"] = pd.to_numeric(ratings_df["Rating"], errors="coerce")
            ratings_df["decade"] = ratings_df["Year"].apply(lambda y: f"{int(y)//10*10}s" if pd.notna(y) else "Unknown")
            st.subheader("Rating distribution")
            fig = px.histogram(ratings_df.dropna(subset=["Rating"]), x="Rating", nbins=10)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Average rating by decade")
            dec = ratings_df.groupby("decade", as_index=False).agg(avg_rating=("Rating", "mean"), count=("Rating", "count"))
            fig = px.bar(dec.sort_values("decade"), x="decade", y="avg_rating", hover_data=["count"])
            st.plotly_chart(fig, use_container_width=True)

        meta_for_analysis = metadata.copy()
        if not meta_for_analysis.empty and not ratings_df.empty:
            merged = ratings_df.merge(prepare_metadata(meta_for_analysis).drop(columns=["Name", "Year"], errors="ignore"), on="movie_id", how="inner")
            if not merged.empty and "genres" in merged.columns:
                rows = []
                for _, row in merged.iterrows():
                    for g in row.get("genres", []) if isinstance(row.get("genres", []), list) else []:
                        rows.append({"genre": g, "Rating": row["Rating"]})
                genre_df = pd.DataFrame(rows)
                if not genre_df.empty:
                    st.subheader("Genre taste profile")
                    gstats = genre_df.groupby("genre", as_index=False).agg(avg_rating=("Rating", "mean"), count=("Rating", "count"))
                    gstats = gstats[gstats["count"] >= 3].sort_values(["avg_rating", "count"], ascending=False)
                    st.dataframe(gstats, use_container_width=True, hide_index=True)

        # Tune watched movies section
        st.subheader("Tune watched movies")
        st.write("Tag watched films with taste labels to steer recommendations. Deliberate tuning here counts more than passive ratings or recommendation feedback.")

        # Rated first so movies that are both watched and rated keep their Rating after dedup.
        rated_movies = data["ratings"].copy()
        watched_movies = data["watched"].copy()
        all_watched = pd.concat([rated_movies, watched_movies], ignore_index=True).drop_duplicates("movie_id")

        if all_watched.empty:
            st.info("No watched movies found.")
        else:
            tuned_movies = all_watched.copy()
            if "Rating" not in tuned_movies.columns:
                tuned_movies["Rating"] = pd.NA
            if not metadata.empty:
                meta_prepared = prepare_metadata(metadata)
                tuned_movies = tuned_movies.merge(
                    meta_prepared[["movie_id", "overview", "genres", "directors", "poster_url"]],
                    on="movie_id",
                    how="left",
                )
            # Attach current feedback labels (only watched_tuning + generic taste labels shown here).
            if not feedback.empty:
                feedback_agg = feedback.groupby("movie_id")["feedback"].agg(list).reset_index()
                tuned_movies = tuned_movies.merge(feedback_agg, on="movie_id", how="left")
            if "feedback" not in tuned_movies.columns:
                tuned_movies["feedback"] = pd.NA

            def _labels(val):
                if isinstance(val, list):
                    return [f for f in val if f in FEEDBACK_LABELS]
                if pd.notna(val) and val in FEEDBACK_LABELS:
                    return [val]
                return []

            tuned_movies["_labels"] = tuned_movies["feedback"].apply(_labels)
            tuned_movies["_tagged"] = tuned_movies["_labels"].apply(bool)

            total_count = len(tuned_movies)
            tagged_count = int(tuned_movies["_tagged"].sum())
            st.caption(f"Tagged {tagged_count} of {total_count} watched films.")

            # Filters
            fcol1, fcol2, fcol3 = st.columns([2, 1, 2])
            search_term = fcol1.text_input("Search movies", key="tune_search")
            only_untagged = fcol2.checkbox("Only untagged", key="tune_untagged")
            rating_range = fcol3.slider(
                "Rating range", 0.5, 5.0, (0.5, 5.0), 0.5, key="tune_rating_range",
                help="Filter by your star rating. Narrow either end to hide films outside the range; unrated films show only at the full range.",
            )

            view = tuned_movies
            if search_term:
                mask = (
                    view["Name"].str.lower().str.contains(search_term.lower(), na=False)
                    | view["Year"].astype(str).str.contains(search_term, na=False)
                )
                view = view[mask]
            if only_untagged:
                view = view[~view["_tagged"]]
            lo, hi = rating_range
            if (lo, hi) != (0.5, 5.0):
                view = view[pd.to_numeric(view["Rating"], errors="coerce").between(lo, hi)]

            # Highest-signal first: untagged before tagged, then by rating desc.
            view = view.assign(_rating_sort=pd.to_numeric(view["Rating"], errors="coerce").fillna(-1))
            view = view.sort_values(["_tagged", "_rating_sort"], ascending=[True, False])

            # Pagination
            PAGE_SIZE = 25
            n_pages = max(1, math.ceil(len(view) / PAGE_SIZE))
            page_num = min(st.session_state.get("tune_page", 0), n_pages - 1)
            pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
            if pcol1.button("◀ Prev", disabled=page_num <= 0, key="tune_prev"):
                st.session_state["tune_page"] = page_num - 1
                st.rerun()
            pcol2.markdown(f"<div style='text-align:center'>Page {page_num + 1} of {n_pages} — {len(view)} films</div>", unsafe_allow_html=True)
            if pcol3.button("Next ▶", disabled=page_num >= n_pages - 1, key="tune_next"):
                st.session_state["tune_page"] = page_num + 1
                st.rerun()

            page_rows = view.iloc[page_num * PAGE_SIZE:(page_num + 1) * PAGE_SIZE]

            # Render page; widget values are read back at save time from session_state.
            page_state = []  # (movie_id, current_labels, widget_key)
            for _, row in page_rows.iterrows():
                mid = row["movie_id"]
                current_feedback = row["_labels"]
                widget_key = f"tunefb_{mid}"
                with st.container():
                    col1, col2, col3 = st.columns([1, 3, 2])
                    with col1:
                        _poster = row.get("poster_url")
                        if pd.notna(_poster) and str(_poster).strip():
                            st.image(str(_poster).strip(), width=80)
                        else:
                            st.write("📽️")
                    with col2:
                        st.write(f"**{row['Name']} ({fmt_year(row.get('Year'))})**")
                        if pd.notna(row.get("overview")):
                            st.caption(row["overview"][:200] + "..." if len(str(row["overview"])) > 200 else str(row["overview"]))
                        genres = row.get("genres", [])
                        if isinstance(genres, list) and genres:
                            st.caption("Genres: " + ", ".join(genres[:3]))
                        rating = row.get("Rating")
                        if pd.notna(rating):
                            st.caption(f"Your rating: {rating}/5")
                    with col3:
                        st.multiselect(
                            "Taste feedback",
                            options=list(FEEDBACK_LABELS.keys()),
                            default=current_feedback,
                            format_func=lambda x: FEEDBACK_LABELS[x]["description"],
                            key=widget_key,
                        )
                page_state.append((mid, current_feedback, widget_key))

            if page_rows.empty:
                st.info("No films match these filters.")
            elif st.button("Save changes on this page", type="primary", key="tune_save_page"):
                changed = 0
                for mid, current_feedback, widget_key in page_state:
                    selected = st.session_state.get(widget_key, current_feedback)
                    new_labels = [l for l in selected if l not in current_feedback]
                    removed_labels = [l for l in current_feedback if l not in selected]
                    for label in new_labels:
                        store_feedback(mid, label, scope="watched_tuning")
                    if removed_labels:
                        remove_feedback(mid, removed_labels)
                    if new_labels or removed_labels:
                        changed += 1
                st.success(f"Saved changes for {changed} film(s).")
                st.rerun()

elif page == "Evaluation":
    st.subheader("Evaluation against historical ratings")
    eval_df, metrics = evaluate_historical_predictions(data, metadata=metadata)
    if not metrics and eval_df.empty:
        st.info("Fetch more TMDb metadata for your rated movies to enable evaluation.")
    elif "error" in metrics:
        st.warning(metrics["error"])
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rated movies with metadata", int(metrics["rated_movies_with_metadata"]))
        m2.metric("Holdout test movies", int(metrics["test_movies"]))
        m3.metric("MAE", f"{metrics['mae']:.2f}")
        m4.metric("Precision@10", f"{metrics['precision_at_10']:.0%}")
        m5, m6, m7 = st.columns(3)
        m5.metric("Recall@25", f"{metrics['recall_at_25']:.0%}")
        m6.metric("NDCG@10", f"{metrics['ndcg_at_10']:.2f}")
        m7.metric("4+ star hits in top 20", int(metrics["top20_4star_hits"]))
        st.caption(f"Similarity/rating correlation: {metrics['correlation']:.2f}")
        try:
            import statsmodels.api  # type: ignore
            fig = px.scatter(eval_df, x="predicted_rating", y="Rating", hover_data=["Name", "Year"], trendline="ols")
        except ModuleNotFoundError:
            fig = px.scatter(eval_df, x="predicted_rating", y="Rating", hover_data=["Name", "Year"])
            st.warning("Optional package `statsmodels` not installed, so the trendline is hidden.")
        st.plotly_chart(fig, use_container_width=True)
        render_grid(eval_df)
        st.write(
            "This holdout test hides about 20% of rated movies, builds a profile from the rest, and checks whether the hidden movies you rated highly rise to the top. "
            "Ranking metrics are more useful than MAE for recommender quality."
        )

elif page == "Curated Weeks":
    st.subheader("Curated movie week")
    cw_mode = st.radio(
        "Curation engine",
        ["Your library (deterministic)", "AI discovery (web search)"],
        horizontal=True,
        help=(
            "Your library scores movies already in your Letterboxd/TMDb data. "
            "AI discovery asks Claude to build a themed week around any film — including "
            "titles you've never logged and brand-new releases verified via web search."
        ),
    )

    if cw_mode == "AI discovery (web search)":
        st.write(
            "Give one anchor film. The model builds a week of 4–5 connected films — a classic, "
            "a thematic match, optionally the same director, a recent (web-verified) title, and a "
            "rewatch — tuned to your Letterboxd taste."
        )

        digest = llm_curator.build_taste_digest(data)
        if digest.loaded:
            mean_txt = f" · avg {digest.mean:.2f}" if digest.mean is not None else ""
            st.caption(f"✓ Taste signal from your library: {digest.count} logged{mean_txt} · {len(digest.top)} top films")
        else:
            st.caption("No local ratings found — the model will curate from the anchor alone.")

        _prov_keys = list(llm_providers.PROVIDERS.keys())
        _default_prov = llm_providers.DEFAULT_PROVIDER if llm_providers.DEFAULT_PROVIDER in _prov_keys else _prov_keys[0]
        prov_col, _ = st.columns([1, 1])
        with prov_col:
            llm_provider = st.selectbox(
                "Provider",
                _prov_keys,
                index=_prov_keys.index(_default_prov),
                format_func=lambda k: llm_providers.PROVIDERS[k].label,
                key="llm_provider",
            )
        _spec = llm_providers.PROVIDERS[llm_provider]
        _prov_ready = _spec.ready
        _search_ok = _spec.supports_search
        st.caption(
            f"Model: `{_spec.model()}` · web verification of recent films: "
            + ("enabled" if _search_ok else "not available on this provider (recent picks unverified)")
        )
        if not _prov_ready:
            st.warning(
                f"No API key for {_spec.label}. Set one of `{'`, `'.join(_spec.api_key_env)}` "
                "in your environment or `.env`."
            )

        anchor_text = st.text_input("Anchor film", placeholder="e.g. Whiplash (2014)", key="llm_anchor")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            llm_seen = st.checkbox("Already seen (= rewatch)", value=True, key="llm_seen")
        with c2:
            llm_director = st.checkbox("Director pick", value=True, key="llm_director")
        with c3:
            llm_era = st.selectbox("Classic pick", ["before 1980", "before 1970"], key="llm_era")
        llm_taste = st.text_area(
            "Taste note (optional)",
            placeholder="e.g. loves procedural/competence dramas, obsession stories; avoid pure comedy.",
            key="llm_taste",
        )

        _era_val = "1970" if "1970" in llm_era else "1980"

        if st.button("Compose the week", disabled=not (anchor_text.strip() and _prov_ready)):
            st.session_state.pop("llm_week", None)
            st.session_state["llm_rejected"] = {}
            with st.spinner("Programming your week…"):
                try:
                    st.session_state["llm_week"] = llm_curator.generate_filmweek(
                        anchor=anchor_text,
                        seen=llm_seen,
                        era=_era_val,
                        with_director=llm_director,
                        taste_note=llm_taste,
                        digest=digest,
                        provider=llm_provider,
                        search_enabled=_search_ok,
                    )
                except Exception as exc:  # noqa: BLE001 — surface any API/parse error to the user
                    st.error(str(exc))

        week = st.session_state.get("llm_week")
        if week:
            _tmdb_key = get_tmdb_api_key()
            _client = TMDbClient(api_key=_tmdb_key, cache_path=cache_path) if _tmdb_key else None
            a = week.get("anchor", {})
            st.divider()
            st.markdown(f"#### ★ Anchor: {a.get('title') or anchor_text}")
            meta_bits = " · ".join(str(x) for x in [a.get("year"), a.get("director")] if x)
            if meta_bits:
                st.caption(meta_bits)
            if a.get("note"):
                st.caption(f"_{a.get('note')}_")

            for pick in week.get("picks", []):
                meta = enrich_llm_pick(pick, _client)
                with st.container(border=True):
                    llm_pick_card(pick, meta, digest)
                    b1, b2, _ = st.columns([1, 1, 3])
                    rejected = st.session_state.setdefault("llm_rejected", {})
                    cat = pick.get("category")

                    def _resim(mode: str, _pick=pick, _cat=cat):
                        shown = [p["title"] for p in week["picks"] if p.get("category") == _cat]
                        excl = list(dict.fromkeys(shown + rejected.get(_cat, [])))
                        with st.spinner("Searching…"):
                            try:
                                new_pick = llm_curator.resimulate_pick(
                                    category=_cat,
                                    anchor_title=a.get("title") or anchor_text,
                                    anchor_year=a.get("year"),
                                    anchor_director=a.get("director", ""),
                                    era=_era_val,
                                    taste_note=llm_taste,
                                    digest=digest,
                                    exclude_titles=excl,
                                    provider=llm_provider,
                                    search_enabled=_search_ok,
                                )
                            except Exception as exc:  # noqa: BLE001
                                st.error(str(exc))
                                return
                        picks = week["picks"]
                        idx = next((i for i, p in enumerate(picks) if p["_id"] == _pick["_id"]), None)
                        if mode == "replace" and idx is not None:
                            rejected.setdefault(_cat, []).append(_pick["title"])
                            picks[idx] = new_pick
                        else:
                            insert_at = max((i for i, p in enumerate(picks) if p.get("category") == _cat), default=len(picks) - 1) + 1
                            picks.insert(insert_at, new_pick)
                        st.rerun()

                    if b1.button("↻ Another", key=f"reroll_{pick['_id']}"):
                        _resim("replace")
                    if b2.button("+ Add one", key=f"add_{pick['_id']}"):
                        _resim("add")
        st.stop()

    st.write("Build an ordered watchlist around one anchor movie using your watched, rated, and watchlist history plus TMDb metadata.")

    if metadata.empty:
        st.info("Fetch TMDb metadata first so the curator can build connected movie weeks.")
    else:
        anchors = anchor_options(metadata, data)
        if anchors.empty:
            st.info("No eligible anchor movies found yet. The curator needs TMDb metadata for movies in your watched, rated, or watchlist data.")
        else:
            source_labels = ["Watched", "Rated", "Watchlist"]
            control_left, control_right = st.columns([2, 1])
            with control_left:
                selected_sources = st.multiselect(
                    "Anchor movie source",
                    source_labels,
                    default=source_labels,
                    help="Choose which parts of your Letterboxd history can supply the anchor movie.",
                )
            with control_right:
                total_movies = st.slider("Number of movies", 3, 14, 7)

            filtered_anchors = anchors.copy()
            if selected_sources:
                selected_source_set = set(selected_sources)
                filtered_anchors = filtered_anchors[
                    filtered_anchors["anchor_sources"].apply(lambda values: bool(set(values) & selected_source_set))
                ].reset_index(drop=True)
            else:
                filtered_anchors = filtered_anchors.iloc[0:0]

            style_col, options_col = st.columns([1, 1])
            with style_col:
                style = st.selectbox("Curation style", CURATION_STYLES, index=0)
            with options_col:
                include_anchor = st.checkbox("Include anchor movie in final list", value=True)
                allow_watched = st.checkbox("Allow watched movies", value=True)
                allow_watchlisted = st.checkbox("Allow watchlisted movies", value=True)

            if filtered_anchors.empty:
                st.warning("No anchor movies match the selected source filters.")
            else:
                anchor_label = st.selectbox("Anchor movie", filtered_anchors["label"].tolist())
                anchor_row = filtered_anchors.loc[filtered_anchors["label"] == anchor_label].iloc[0]

                anchor_meta_left, anchor_meta_right = st.columns([1, 3])
                with anchor_meta_left:
                    _pu = anchor_row.get("poster_url")
                    if pd.notna(_pu) and str(_pu).strip():
                        st.image(str(_pu).strip(), use_container_width=True)
                with anchor_meta_right:
                    st.caption("Anchor sources: " + (anchor_row.get("source_labels") or "Unknown"))
                    anchor_genres = anchor_row.get("genres", []) if isinstance(anchor_row.get("genres"), list) else []
                    anchor_moods = anchor_row.get("moods", []) if isinstance(anchor_row.get("moods"), list) else []
                    if anchor_genres:
                        st.caption("Genres: " + ", ".join(anchor_genres[:4]))
                    if anchor_moods:
                        st.caption("Moods: " + ", ".join(anchor_moods[:4]))
                    if anchor_row.get("overview"):
                        st.write(anchor_row.get("overview"))

                try:
                    curated = build_curated_list(
                        anchor_movie_id=str(anchor_row["movie_id"]),
                        data=data,
                        metadata=metadata,
                        total_movies=int(total_movies),
                        style=style,
                        allow_watched=allow_watched,
                        allow_watchlisted=allow_watchlisted,
                        include_anchor=include_anchor,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    curated = pd.DataFrame()

                if curated.empty:
                    st.warning("The curator could not build a movie week from the current filters. Try allowing watched or watchlisted movies, or choose another anchor.")
                else:
                    if len(curated) < total_movies:
                        st.info(f"Built {len(curated)} movies instead of {total_movies} because the filtered candidate pool ran out.")

                    intensity_map = {
                        "Context / influence": 2,
                        "Thematic setup": 4,
                        "Anchor movie": 7,
                        "Director / actor connection": 5,
                        "Intensifier": 8,
                        "Contrast / decompression": 3,
                        "Afterglow / reflection": 2,
                        "Companion film": 5,
                    }
                    curve = curated[["day", "role"]].copy()
                    curve["intensity"] = curve["role"].map(intensity_map).fillna(5)
                    st.caption("Flow across the week")
                    st.line_chart(curve.set_index("day")["intensity"], use_container_width=True)

                    for _, row in curated.iterrows():
                        with st.container(border=True):
                            curated_week_card(row)

                    if use_database:
                        st.divider()
                        save_col, _ = st.columns([2, 1])
                        with save_col:
                            save_label = st.text_input("Week label (optional)", placeholder="e.g. Tarkovsky deep dive", key="curated_week_label")
                            if st.button("Save this curated week"):
                                week_id = save_curated_week(
                                    anchor_movie_id=str(anchor_row["movie_id"]),
                                    anchor_name=str(anchor_row.get("Name", anchor_row["movie_id"])),
                                    style=style,
                                    curated_df=curated,
                                    label=save_label,
                                    db_path=db_path,
                                )
                                st.success(f"Saved as week #{week_id}.")

                with st.expander("Load a saved curated week"):
                    if not use_database:
                        st.caption("Saved weeks require the SQLite backend. Build the database first.")
                    else:
                        saved_weeks = load_curated_weeks(db_path=db_path)
                        if saved_weeks.empty:
                            st.caption("No saved weeks yet.")
                        else:
                            saved_weeks["display"] = saved_weeks.apply(
                                lambda r: f"#{r['id']} — {r['anchor_name']} ({r['style']}, {r['total_movies']} films) {r['created_at'][:10]}"
                                + (f" — {r['label']}" if r.get("label") else ""),
                                axis=1,
                            )
                            sel_week_label = st.selectbox("Select saved week", saved_weeks["display"].tolist(), key="load_curated_select")
                            if st.button("Load selected week"):
                                sel_id = int(saved_weeks.loc[saved_weeks["display"] == sel_week_label, "id"].iloc[0])
                                loaded = load_curated_week(sel_id, db_path=db_path)
                                if not loaded.empty:
                                    st.subheader("Loaded curated week")
                                    for _, lrow in loaded.iterrows():
                                        with st.container(border=True):
                                            curated_week_card(lrow)

elif page == "Reflection":
    _tmdb_key = get_tmdb_api_key()
    _reflection_client = TMDbClient(api_key=_tmdb_key, cache_path=cache_path) if _tmdb_key else None
    render_reflection_panel(
        data, metadata, tmdb_client=_reflection_client,
        db_path=db_path, use_database=use_database, cache_path=cache_path,
    )

else:  # "Data & Sync"
    st.subheader("Data & Sync")
    st.caption("TMDb metadata caching, Letterboxd sync, and the SQLite backend — set these up once, then check back occasionally.")

    dm1, dm2, dm3, dm4 = st.columns(4)
    dm1.metric("Cached movies", cached_count)
    dm2.metric("Known-profile cached", known_count)
    dm3.metric("TMDb matches", found_count)
    dm4.metric("Backend", "SQLite" if use_database else "CSV/JSON")

    tab_tmdb, tab_sync, tab_db = st.tabs(["TMDb metadata", "Letterboxd sync", "SQLite database"])

    with tab_tmdb:
        st.text_input(
            "TMDb API key",
            value=os.getenv("TMDB_API_KEY", ""),
            type="password",
            key="tmdb_api_key_input",
            help="Optional. You can also set TMDB_API_KEY in your environment or Streamlit secrets.",
        )
        with st.expander("Enrich known Letterboxd movies", expanded=True):
            st.write("Repeated runs skip already cached movies unless refresh is enabled.")
            limit = st.number_input("Uncached movies to fetch this run", min_value=1, max_value=max(1, int(len(all_movies))), value=min(50, int(len(all_movies))), step=25)
            force = st.checkbox("Refresh existing cached movies", value=False)
            if st.button("Fetch TMDb metadata"):
                key = get_tmdb_api_key()
                if not key:
                    st.error("Add a TMDb API key first.")
                else:
                    client = TMDbClient(api_key=key, cache_path=cache_path)
                    with st.spinner("Fetching and caching TMDb metadata..."):
                        result = enrich_movies(all_movies, client=client, limit=int(limit), force=force)
                        if use_database:
                            import_tmdb_cache(cache_path=cache_path, db_path=db_path)
                    st.success(f"Fetched or refreshed {len(result)} movies. Refreshing recommendations.")
                    st.rerun()

        with st.expander("Discover new outside-watchlist candidates"):
            st.write("Uses TMDb recommendations and similar-movie endpoints from your high-rated cached movies.")
            per_seed = st.number_input("Candidates per seed", min_value=2, max_value=20, value=8, step=2)
            seed_limit = st.number_input("High-rated seed movies", min_value=1, max_value=100, value=25, step=5)
            if st.button("Discover from favorites"):
                key = get_tmdb_api_key()
                if not key:
                    st.error("Add a TMDb API key first.")
                else:
                    meta = prepare_metadata(metadata)
                    ratings = data["ratings"].copy()
                    ratings["Rating"] = pd.to_numeric(ratings.get("Rating"), errors="coerce")
                    favorite_ids = set(ratings.loc[ratings["Rating"] >= 4.0, "movie_id"].dropna()) | set(data["likes"].get("movie_id", pd.Series(dtype=str)).dropna())
                    favorite_meta = meta[meta["movie_id"].isin(favorite_ids)].copy()
                    if "tmdb_popularity" in favorite_meta.columns:
                        favorite_meta = favorite_meta.sort_values("tmdb_popularity", ascending=False, na_position="last")
                    if favorite_meta.empty:
                        st.warning("Cache TMDb metadata for rated/liked movies first.")
                    else:
                        client = TMDbClient(api_key=key, cache_path=cache_path)
                        with st.spinner("Discovering and caching outside-watchlist candidates..."):
                            discovered = discover_movies_from_favorites(favorite_meta, client=client, per_seed=int(per_seed), seed_limit=int(seed_limit))
                            if use_database:
                                import_tmdb_cache(cache_path=cache_path, db_path=db_path)
                        st.success(f"Discovered/cached {len(discovered)} candidate movies. Refreshing recommendations.")
                        st.rerun()

        with st.expander("Command-line enrichment"):
            st.code("export TMDB_API_KEY='your_key_here'\nexport LETTERBOXD_USERNAME='your_username'\npython sync_letterboxd.py $LETTERBOXD_USERNAME --status\npython enrich_tmdb.py --limit 100\npython enrich_tmdb.py", language="bash")

    with tab_sync:
        sync_stat = sync_status()
        st.caption(f"RSS events: {sync_stat.get('rss_events', 0)} | Last sync: {sync_stat.get('last_sync_at', 'never')}")

        with st.expander("Sync recent activity from RSS", expanded=True):
            st.write("RSS updates recent watches, diary entries, rewatches, and ratings that appear in your public activity feed. Use a fresh export for full watchlist state and old rating edits.")
            lb_username = st.text_input("Letterboxd username or RSS URL", value=os.getenv("LETTERBOXD_USERNAME", ""), help="Example: bslinky or https://letterboxd.com/bslinky/rss/")
            if st.button("Sync Letterboxd RSS"):
                if not lb_username:
                    st.error("Add your Letterboxd username or RSS URL first.")
                else:
                    with st.spinner("Fetching Letterboxd RSS..."):
                        result = sync_rss(lb_username)
                    if use_database:
                        with st.spinner("Applying synced events to database..."):
                            apply_rss_overlays_to_db(db_path=db_path)
                    new_ev = result.get("new_events", 0)
                    st.success(
                        f"Fetched {result.get('fetched_events', 0)} events; "
                        f"{new_ev} new."
                        + (" Database updated." if use_database else "")
                    )
                    # Persist username to .env so auto-sync works next session.
                    if _DOTENV_AVAILABLE and lb_username != os.getenv("LETTERBOXD_USERNAME", ""):
                        try:
                            _env_file = find_dotenv(usecwd=True) or ".env"
                            set_key(_env_file, "LETTERBOXD_USERNAME", lb_username)
                        except Exception:
                            pass
                    st.rerun()

        with st.expander("Replace with fresh Letterboxd export"):
            st.write("Use this when you want authoritative updates for watchlist removals/additions, old rating edits, deleted ratings, and historical changes not present in RSS.")
            uploaded_export = st.file_uploader("Upload latest Letterboxd export zip", type=["zip"])
            if uploaded_export is not None and st.button("Install uploaded export"):
                export_zip.parent.mkdir(parents=True, exist_ok=True)
                export_zip.write_bytes(uploaded_export.getbuffer())
                # Force re-extraction next run.
                if Path("data/letterboxd").exists():
                    import shutil
                    shutil.rmtree(Path("data/letterboxd"))
                if use_database:
                    with st.spinner("Rebuilding database from new export..."):
                        rebuild_database(export_zip=export_zip, cache_path=cache_path, db_path=db_path)
                    st.success("Installed new Letterboxd export and rebuilt database.")
                else:
                    st.success("Installed latest Letterboxd export. Refreshing data.")
                st.rerun()

        st.divider()
        st.caption("Sync overlay detail")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("RSS events", int(sync_stat.get("rss_events", 0)))
        s2.metric("Watched overlay", int(sync_stat.get("watched_overlay", 0)))
        s3.metric("Ratings overlay", int(sync_stat.get("ratings_overlay", 0)))
        s4.metric("Diary overlay", int(sync_stat.get("diary_overlay", 0)))
        st.info("RSS sync is incremental and best for recent activity. A fresh Letterboxd export is still the source of truth for complete watchlist state, old rating edits, deleted ratings, and historical backfills.")
        for label, path in {
            "Recent RSS events": Path("data/sync/rss_events.csv"),
            "Rating changes overlay": Path("data/sync/ratings_overlay.csv"),
            "Watched overlay": Path("data/sync/watched_overlay.csv"),
            "Diary / rewatches overlay": Path("data/sync/diary_overlay.csv"),
        }.items():
            with st.expander(label):
                if path.exists():
                    st.dataframe(pd.read_csv(path).tail(100), use_container_width=True, hide_index=True)
                else:
                    st.caption("No data yet.")

    with tab_db:
        st.write("Use SQLite as the app backend for analysis, history, rating changes, rewatches, metadata, and feedback.")
        if st.button("Build / refresh database from local files"):
            with st.spinner("Importing Letterboxd export, TMDb cache, and feedback into SQLite..."):
                result = rebuild_database(export_zip=export_zip, cache_path=cache_path, db_path=db_path)
            st.success("Database rebuilt.")
            st.json(result)
            st.rerun()
        st.caption(f"Database path: {db_path}")

        st.divider()
        db_status = database_status(db_path)
        if not db_status.get("exists"):
            st.warning("Database does not exist yet. Use the button above to build it from your local files.")
        else:
            st.json(db_status)
            st.write("The app reads from SQLite when `data/movie_recommender.sqlite` exists. CSV/JSON/RSS remain ingestion sources.")
            if st.button("Import latest TMDb cache into database"):
                count = import_tmdb_cache(cache_path=cache_path, db_path=db_path)
                st.success(f"Imported {count} metadata rows.")
                st.rerun()
            if st.button("Import latest Letterboxd export into database"):
                result = import_letterboxd_export(export_zip=export_zip, db_path=db_path)
                st.success("Imported latest export.")
                st.json(result)
                st.rerun()
