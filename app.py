from datetime import datetime, timezone, timedelta
from pathlib import Path
import os

import pandas as pd
import plotly.express as px
import streamlit as st
import re

try:
    from dotenv import load_dotenv, set_key, find_dotenv
    load_dotenv()
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

from curator import CURATION_STYLES, anchor_options, build_curated_list
from recommender import (
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

        /* ---- Buttons ---- */
        .stButton > button {
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
            font-weight: 500;
            transition: all 0.16s ease;
        }
        .stButton > button:hover {
            border-color: rgba(201,162,39,0.6);
            color: #F5E6A8;
            background: rgba(201,162,39,0.10);
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

export_zip = Path("data/letterboxd_export.zip")
if not export_zip.exists():
    st.error("Put your Letterboxd export zip at data/letterboxd_export.zip")
    st.stop()

db_path = Path("data/movie_recommender.sqlite")
use_database = db_path.exists()

# --- Startup auto-sync ---
# Runs once per session if LETTERBOXD_USERNAME is set and last sync was > 1 hour ago.
# Updates overlay CSVs (and SQLite if active) before data loads, so the session starts fresh.
if "auto_synced_this_session" not in st.session_state:
    st.session_state.auto_synced_this_session = False

_lb_auto_user = os.getenv("LETTERBOXD_USERNAME", "")
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

movie_frames = [
    data["ratings"][["Name", "Year"]],
    data["watched"][["Name", "Year"]],
    data["watchlist"][["Name", "Year"]],
    data["likes"][["Name", "Year"]],
]
if not data["lists"].empty:
    movie_frames.append(data["lists"][["Name", "Year"]])
all_movies = pd.concat(movie_frames, ignore_index=True).drop_duplicates()

if _auto_new_events > 0:
    st.toast(f"Auto-synced Letterboxd: {_auto_new_events} new events added.")

st.sidebar.header("TMDb metadata")
api_key_input = st.sidebar.text_input(
    "TMDb API key",
    value=os.getenv("TMDB_API_KEY", ""),
    type="password",
    help="Optional. You can also set TMDB_API_KEY in your environment or Streamlit secrets.",
)
cache_path = Path("data/tmdb_cache.json")
if use_database:
    metadata = load_metadata_from_db(db_path)
    metadata_known = metadata
    feedback = load_feedback_from_db(db_path)
else:
    metadata = metadata_from_cache(None, cache_path=cache_path, include_all=True)
    metadata_known = metadata_from_cache(all_movies, cache_path=cache_path)
    feedback = load_feedback()

cached_count = len(metadata) if not metadata.empty else 0
known_count = len(metadata_known) if not metadata_known.empty else 0
found_count = int(metadata.get("tmdb_found", pd.Series(dtype=bool)).fillna(False).sum()) if not metadata.empty else 0
st.sidebar.metric("Cached movies", cached_count)
st.sidebar.metric("Known-profile cached", known_count)
st.sidebar.metric("TMDb matches", found_count)
st.sidebar.caption("SQLite backend: " + ("on" if use_database else "off - using CSV/JSON files"))

st.sidebar.header("Database")
with st.sidebar.expander("SQLite database"):
    st.write("Use SQLite as the app backend for analysis, history, rating changes, rewatches, metadata, and feedback.")
    if st.button("Build / refresh database from local files"):
        with st.spinner("Importing Letterboxd export, TMDb cache, and feedback into SQLite..."):
            result = rebuild_database(export_zip=export_zip, cache_path=cache_path, db_path=db_path)
        st.success("Database rebuilt.")
        st.json(result)
        st.rerun()
    st.caption(f"Database path: {db_path}")

st.sidebar.header("Letterboxd sync")
status = sync_status()
st.sidebar.caption(f"RSS events: {status.get('rss_events', 0)} | Last sync: {status.get('last_sync_at', 'never')}")
with st.sidebar.expander("Sync recent activity from RSS"):
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

with st.sidebar.expander("Replace with fresh Letterboxd export"):
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

with st.sidebar.expander("Enrich known Letterboxd movies"):
    st.write("Repeated runs skip already cached movies unless refresh is enabled.")
    limit = st.number_input("Uncached movies to fetch this run", min_value=1, max_value=max(1, int(len(all_movies))), value=min(50, int(len(all_movies))), step=25)
    force = st.checkbox("Refresh existing cached movies", value=False)
    if st.button("Fetch TMDb metadata"):
        key = api_key_input or os.getenv("TMDB_API_KEY")
        if not key:
            st.error("Add a TMDb API key first.")
        else:
            client = TMDbClient(api_key=key, cache_path=cache_path)
            with st.spinner("Fetching and caching TMDb metadata..."):
                result = enrich_movies(all_movies, client=client, limit=int(limit), force=force)
            st.success(f"Fetched or refreshed {len(result)} movies. Refreshing recommendations.")
            st.rerun()

with st.sidebar.expander("Discover new outside-watchlist candidates"):
    st.write("Uses TMDb recommendations and similar-movie endpoints from your high-rated cached movies.")
    per_seed = st.number_input("Candidates per seed", min_value=2, max_value=20, value=8, step=2)
    seed_limit = st.number_input("High-rated seed movies", min_value=1, max_value=100, value=25, step=5)
    if st.button("Discover from favorites"):
        key = api_key_input or os.getenv("TMDB_API_KEY")
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
                st.success(f"Discovered/cached {len(discovered)} candidate movies. Refreshing recommendations.")
                st.rerun()

ALL_MOODS = ["Tense", "Emotional", "Gritty", "Exciting", "Imaginative", "Light", "Reflective"]

mode_label = st.sidebar.radio(
    "Recommendation source",
    ["My watchlist", "Not on my watchlist"],
    help="Outside-watchlist recommendations use your lists plus TMDb-discovered cached records, excluding watched/rated/watchlisted movies.",
)
mode = "outside_watchlist" if mode_label == "Not on my watchlist" else "watchlist"

filter_values_preview = available_filter_values(pd.DataFrame())
taste_mode = st.sidebar.selectbox("Taste mode", filter_values_preview.get("taste_modes", ["Balanced"]), index=0)

with st.sidebar.expander("Scoring weights"):
    st.caption("Drag to change how much each signal pulls the final score.")
    content_weight = st.slider("Taste similarity", 0.0, 3.0, 1.0, 0.25, help="How strongly TF-IDF content similarity to your high-rated films affects the score.")
    entity_weight = st.slider("Director / cast influence", 0.0, 3.0, 1.0, 0.25, help="How strongly a shared director, writer, or cast member you've rated highly affects the score.")
    list_weight = st.slider("List signals", 0.0, 3.0, 1.0, 0.25, help="How much being on your curated lists counts.")
    anchor_weight = st.slider("Anchor influence", 0.0, 3.0, 1.0, 0.25, help="How strongly the film you anchor on (Recommendations page) pulls similar candidates up.")
score_weights = {"content": content_weight, "entity": entity_weight, "list": list_weight, "anchor": anchor_weight}

page = st.sidebar.radio("Page", ["Tonight's Pick", "Recommendations", "Analysis", "Evaluation", "Curated Weeks", "Database", "Sync status"])

recs, decade_prefs = build_recommendations(data, metadata=metadata, mode=mode, feedback=feedback, taste_mode=taste_mode, score_weights=score_weights)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Rated", len(data["ratings"]))
col2.metric("Watched", len(data["watched"]))
col3.metric("Watchlist", len(data["watchlist"]))
col4.metric("Custom list entries", len(data["lists"]))
col5.metric("Metadata coverage", f"{known_count}/{len(all_movies)}")


def store_feedback(movie_id: str, feedback_value: str) -> None:
    if use_database:
        save_feedback_to_db(movie_id, feedback_value, db_path=db_path)
    else:
        save_feedback(movie_id, feedback_value)


def remove_feedback(movie_id: str, labels: list) -> None:
    if use_database:
        remove_feedback_from_db(movie_id, labels, db_path=db_path)
    else:
        remove_feedback_from_csv(movie_id, labels)


def poster_card(row: pd.Series, idx: int) -> None:
    title = f"{row.get('Name', '')} ({row.get('Year', '')})"
    _pu = row.get("poster_url")
    if pd.notna(_pu) and str(_pu).strip():
        st.image(str(_pu).strip(), use_container_width=True)
    else:
        st.info("No poster")
    st.markdown(f"**{title}**")
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
        st.markdown(f"### Day {int(row.get('day', 0))}: {row.get('Name', '')} ({row.get('Year', '')})")
        st.caption(f"{row.get('role', '')} | {row.get('role_description', '')}")
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


def render_score_breakdown(row: pd.Series, score_weights: dict) -> None:
    content_w = float(score_weights.get("content", 1.0))
    entity_w = float(score_weights.get("entity", 1.0))
    list_w = float(score_weights.get("list", 1.0))

    list_contrib = float(row.get("list_contribution", 0) or 0)
    heuristic = float(row.get("heuristic_score", 3.0) or 3.0)
    base_delta = heuristic - list_contrib - 3.0  # decade + recency above the 3.0 baseline

    components = [
        ("Decade & recency", base_delta),
        ("List signals", list_contrib * list_w),
        ("Taste similarity", float(row.get("content_score", 0) or 0) * content_w),
        ("Feedback", float(row.get("feedback_score", 0) or 0)),
        ("Taste mode", float(row.get("taste_mode_score", 0) or 0)),
        ("Dir / Cast affinity", float(row.get("entity_score", 0) or 0) * entity_w),
        ("Anchor match", float(row.get("anchor_score", 0) or 0)),
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
        tonight_time_label = st.radio("Time available", list(_time_options.keys()), horizontal=True, key="tonight_time")
        tonight_runtime = _time_options[tonight_time_label]

    with st.expander("Not in the mood for...", expanded=False):
        tonight_avoid = st.multiselect("Avoid these moods", ALL_MOODS, key="tonight_avoid_moods")

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
            st.markdown(f"## {pick.get('Name', '')} ({pick.get('Year', '')})")
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
            with st.expander("Why this?", expanded=True):
                render_reasons(str(pick.get("why_details") or pick.get("why", "")))
                if pick.get("overview"):
                    st.write(str(pick["overview"]))

        with st.expander("Score breakdown", expanded=False):
            render_score_breakdown(pick, score_weights)

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
    with st.expander("Anchor on a film", expanded=False):
        st.caption("Pick a film you love and the engine will boost candidates most similar to it.")
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
                if anchor_choice != "— none —":
                    chosen_idx = anchor_labels.index(anchor_choice) - 1
                    anchor_movie_id = str(anchor_pool.iloc[chosen_idx]["movie_id"])
                    st.caption(f"Boosting candidates similar to: **{anchor_choice}**")

    # --- Mood avoidance ---
    with st.expander("Not in the mood for...", expanded=False):
        st.caption("Temporarily penalise these moods in this session. No permanent feedback saved.")
        avoid_moods = st.multiselect("Avoid tonight", ALL_MOODS)

    # Re-run scoring if anchor or mood avoidance is active
    if anchor_movie_id or avoid_moods:
        from recommender import build_recommendations as _build_recs
        recs, decade_prefs = _build_recs(
            data, metadata=metadata, mode=mode, feedback=feedback, taste_mode=taste_mode,
            score_weights=score_weights, anchor_movie_id=anchor_movie_id, avoid_moods=avoid_moods,
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

    view = st.radio("View", ["Poster cards", "Table"], horizontal=True)
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
        show_cols = ["Name", "Year", "score", "heuristic_score", "content_score", "feedback_score", "taste_mode_score", "entity_score", "anchor_score", "mood_penalty", "why", "Letterboxd URI"]
        show_cols += [c for c in ["genres", "moods", "runtime", "languages", "directors", "cast", "keywords", "tmdb_url", "discovered_from"] if c in filtered.columns]
        st.dataframe(filtered[show_cols].head(100), use_container_width=True, hide_index=True)

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
                render_score_breakdown(full_row, score_weights)

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
        st.write("Use these labels to provide richer taste signals for better recommendations. This feedback is stronger than passive ratings.")

        # Get watched/rated movies
        watched_movies = data["watched"].copy()
        rated_movies = data["ratings"].copy()
        all_watched = pd.concat([watched_movies, rated_movies], ignore_index=True).drop_duplicates("movie_id")

        if all_watched.empty:
            st.info("No watched movies found.")
        else:
            # Merge with metadata and feedback
            tuned_movies = all_watched.copy()
            if not metadata.empty:
                meta_prepared = prepare_metadata(metadata)
                tuned_movies = tuned_movies.merge(
                    meta_prepared[["movie_id", "overview", "genres", "directors", "poster_url"]],
                    on="movie_id",
                    how="left"
                )
            if not feedback.empty:
                feedback_agg = feedback.groupby("movie_id")["feedback"].agg(list).reset_index()
                tuned_movies = tuned_movies.merge(feedback_agg, on="movie_id", how="left")

            # Add search/filter
            search_term = st.text_input("Search movies", key="tune_search")
            if search_term:
                mask = (
                    tuned_movies["Name"].str.lower().str.contains(search_term.lower(), na=False) |
                    tuned_movies["Year"].astype(str).str.contains(search_term, na=False)
                )
                tuned_movies = tuned_movies[mask]

            # Show movies with feedback controls
            st.write(f"Showing {len(tuned_movies)} movies")
            for idx, row in tuned_movies.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([1, 3, 2])
                    with col1:
                        _poster = row.get("poster_url")
                        if pd.notna(_poster) and str(_poster).strip():
                            st.image(str(_poster).strip(), width=80)
                        else:
                            st.write("📽️")
                    with col2:
                        st.write(f"**{row['Name']} ({row['Year']})**")
                        if pd.notna(row.get("overview")):
                            st.caption(row["overview"][:200] + "..." if len(str(row["overview"])) > 200 else str(row["overview"]))
                        genres = row.get("genres", [])
                        if isinstance(genres, list) and genres:
                            st.caption("Genres: " + ", ".join(genres[:3]))
                        rating = row.get("Rating")
                        if pd.notna(rating):
                            st.caption(f"Your rating: {rating}/5")
                    with col3:
                        current_feedback = row.get("feedback", [])
                        if not isinstance(current_feedback, list):
                            current_feedback = [current_feedback] if pd.notna(current_feedback) else []
                        current_feedback = [f for f in current_feedback if f in FEEDBACK_LABELS]

                        selected = st.multiselect(
                            "Taste feedback",
                            options=list(FEEDBACK_LABELS.keys()),
                            default=current_feedback,
                            format_func=lambda x: FEEDBACK_LABELS[x]["description"],
                            key=f"feedback_{row['movie_id']}_{idx}",
                        )

                        new_labels = [l for l in selected if l not in current_feedback]
                        removed_labels = [l for l in current_feedback if l not in selected]
                        if new_labels or removed_labels:
                            if st.button("Save feedback", key=f"save_{row['movie_id']}_{idx}"):
                                for label in new_labels:
                                    store_feedback(row["movie_id"], label)
                                if removed_labels:
                                    remove_feedback(row["movie_id"], removed_labels)
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
        st.dataframe(eval_df, use_container_width=True, hide_index=True)
        st.write(
            "This holdout test hides about 20% of rated movies, builds a profile from the rest, and checks whether the hidden movies you rated highly rise to the top. "
            "Ranking metrics are more useful than MAE for recommender quality."
        )

elif page == "Curated Weeks":
    st.subheader("Curated movie week")
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

elif page == "Database":
    st.subheader("SQLite database")
    status = database_status(db_path)
    if not status.get("exists"):
        st.warning("Database does not exist yet. Use the sidebar button to build it from your local files.")
    else:
        st.json(status)
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

else:
    st.subheader("Letterboxd sync status")
    status = sync_status()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("RSS events", int(status.get("rss_events", 0)))
    s2.metric("Watched overlay", int(status.get("watched_overlay", 0)))
    s3.metric("Ratings overlay", int(status.get("ratings_overlay", 0)))
    s4.metric("Diary overlay", int(status.get("diary_overlay", 0)))
    st.write("Last sync:", status.get("last_sync_at", "never"))
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

with st.expander("Command-line enrichment"):
    st.code("export TMDB_API_KEY='your_key_here'\nexport LETTERBOXD_USERNAME='your_username'\npython sync_letterboxd.py $LETTERBOXD_USERNAME --status\npython enrich_tmdb.py --limit 100\npython enrich_tmdb.py", language="bash")
