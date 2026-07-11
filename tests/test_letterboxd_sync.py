"""Tests for the Letterboxd RSS sync (parsing, overlays, and the list-filtering guard)."""
from __future__ import annotations

import pandas as pd
import pytest

from letterboxd_sync import (
    _parse_rating,
    _title_year_from_title,
    apply_sync_overlays,
    merge_rss_events,
    parse_rss_items,
    sync_status,
    username_to_rss_url,
)


def _rss(items: str) -> str:
    return (
        '<rss xmlns:letterboxd="https://letterboxd.com" '
        'xmlns:tmdb="https://www.themoviedb.org"><channel>'
        + items
        + "</channel></rss>"
    )


DIARY_ITEM = """
<item>
  <title>Inception, 2010 - ★★★★½</title>
  <link>https://letterboxd.com/bob/film/inception/</link>
  <guid>letterboxd-review-1</guid>
  <pubDate>Wed, 01 Jan 2020 12:00:00 +0000</pubDate>
  <letterboxd:filmTitle>Inception</letterboxd:filmTitle>
  <letterboxd:filmYear>2010</letterboxd:filmYear>
  <letterboxd:watchedDate>2020-01-01</letterboxd:watchedDate>
  <letterboxd:memberRating>4.5</letterboxd:memberRating>
  <letterboxd:rewatch>No</letterboxd:rewatch>
  <tmdb:movieId>27205</tmdb:movieId>
</item>
"""

# A member's *list* entry — must be filtered out (guid + /list/ link).
LIST_ITEM = """
<item>
  <title>My Favourite Heist Films</title>
  <link>https://letterboxd.com/bob/list/my-favourite-heist-films/</link>
  <guid>letterboxd-list-456</guid>
  <pubDate>Wed, 01 Jan 2020 12:00:00 +0000</pubDate>
</item>
"""


# --- username / URL normalisation -------------------------------------------

def test_username_to_rss_url_from_plain_username():
    assert username_to_rss_url("bob") == "https://letterboxd.com/bob/rss/"


def test_username_to_rss_url_from_profile_url():
    assert username_to_rss_url("https://letterboxd.com/bob/") == "https://letterboxd.com/bob/rss/"


def test_username_to_rss_url_already_rss():
    assert username_to_rss_url("https://letterboxd.com/bob/rss/") == "https://letterboxd.com/bob/rss/"


def test_username_to_rss_url_empty_raises():
    with pytest.raises(ValueError):
        username_to_rss_url("   ")


# --- rating / title parsing --------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("4.5", 4.5),
    ("★★★", 3.0),
    ("★★★½", 3.5),
    ("", None),
    ("not a rating", None),
])
def test_parse_rating(raw, expected):
    assert _parse_rating(raw) == expected


def test_title_year_extraction():
    assert _title_year_from_title("Inception, 2010 - ★★★★½") == ("Inception", 2010)
    assert _title_year_from_title("No Year Film") == ("No Year Film", None)


# --- parse_rss_items ---------------------------------------------------------

def test_parse_rss_items_reads_diary_entry():
    df = parse_rss_items(_rss(DIARY_ITEM))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["Name"] == "Inception"
    assert int(row["Year"]) == 2010
    assert row["Rating"] == 4.5
    assert row["movie_id"] == "inception (2010)"
    assert row["Watched Date"] == "2020-01-01"


def test_parse_rss_items_filters_out_list_entries():
    """Regression: the per-user RSS feed mixes in the member's lists; they must not
    be ingested as phantom watched/rated movies."""
    df = parse_rss_items(_rss(DIARY_ITEM + LIST_ITEM))
    assert len(df) == 1
    assert "my favourite heist films" not in set(df["Name"].str.lower())
    assert df.iloc[0]["Name"] == "Inception"


def test_parse_rss_items_never_scrapes_rating_from_bare_title_number():
    # A film titled with a number and no star glyphs must not invent a rating.
    item = """
    <item>
      <title>1917</title>
      <link>https://letterboxd.com/bob/film/1917/</link>
      <guid>letterboxd-watch-9</guid>
      <letterboxd:filmTitle>1917</letterboxd:filmTitle>
      <letterboxd:filmYear>2019</letterboxd:filmYear>
    </item>
    """
    df = parse_rss_items(_rss(item))
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["Rating"])


def test_parse_rss_items_empty_feed():
    assert parse_rss_items(_rss("")).empty


# --- merge + overlays --------------------------------------------------------

def test_merge_rss_events_writes_overlays_and_dedupes(tmp_path):
    events = parse_rss_items(_rss(DIARY_ITEM))
    first = merge_rss_events(events, sync_dir=tmp_path)
    assert first["new_events"] == 1
    assert (tmp_path / "watched_overlay.csv").exists()
    assert (tmp_path / "ratings_overlay.csv").exists()

    # Re-merging the same event must not create a duplicate.
    second = merge_rss_events(events, sync_dir=tmp_path)
    assert second["total_events"] == 1

    status = sync_status(sync_dir=tmp_path)
    assert status["watched_overlay"] == 1
    assert status["ratings_overlay"] == 1


def test_apply_sync_overlays_merges_rating_into_base(tmp_path):
    merge_rss_events(parse_rss_items(_rss(DIARY_ITEM)), sync_dir=tmp_path)
    base = {
        "ratings": pd.DataFrame(columns=["Name", "Year", "Rating", "movie_id"]),
        "watched": pd.DataFrame(columns=["Name", "Year", "movie_id"]),
        "diary": pd.DataFrame(columns=["Name", "Year", "Watched Date", "movie_id"]),
        "watchlist": pd.DataFrame(columns=["Name", "Year", "movie_id"]),
        "likes": pd.DataFrame(columns=["Name", "Year", "movie_id"]),
        "lists": pd.DataFrame(),
    }
    out = apply_sync_overlays(base, sync_dir=tmp_path)
    assert "inception (2010)" in set(out["ratings"]["movie_id"])
    assert out["ratings"].loc[out["ratings"]["movie_id"] == "inception (2010)", "Rating"].iloc[0] == 4.5
    assert "inception (2010)" in set(out["watched"]["movie_id"])
