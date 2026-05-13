from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from recommender import ensure_export_dir, load_letterboxd, prepare_metadata
from tmdb_client import TMDbClient, discover_movies_from_favorites, enrich_movies, metadata_from_cache


def all_letterboxd_movies(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [
        data["ratings"][["Name", "Year"]],
        data["watched"][["Name", "Year"]],
        data["watchlist"][["Name", "Year"]],
        data["likes"][["Name", "Year"]],
    ]
    if not data["lists"].empty:
        frames.append(data["lists"][["Name", "Year"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache TMDb metadata for your Letterboxd movies.")
    parser.add_argument("--export-zip", default="data/letterboxd_export.zip")
    parser.add_argument("--export-dir", default="data/letterboxd")
    parser.add_argument("--cache", default="data/tmdb_cache.json")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for known Letterboxd movies.")
    parser.add_argument("--force", action="store_true", help="Refresh movies already in the cache.")
    parser.add_argument("--discover", action="store_true", help="Also discover outside-watchlist candidates from high-rated cached movies.")
    parser.add_argument("--seed-limit", type=int, default=25, help="Number of high-rated seed movies for discovery.")
    parser.add_argument("--per-seed", type=int, default=8, help="Number of TMDb similar/recommended movies per seed endpoint.")
    args = parser.parse_args()

    client = TMDbClient.from_env(cache_path=args.cache)
    if client is None:
        raise SystemExit("Set TMDB_API_KEY before running enrichment.")

    data = apply_sync_overlays(load_letterboxd(ensure_export_dir(args.export_zip, args.export_dir)))
    movies = all_letterboxd_movies(data)

    print(f"Enriching {len(movies) if args.limit is None else min(args.limit, len(movies))} uncached known movies...")
    result = enrich_movies(movies, client=client, limit=args.limit, force=args.force)
    found = int(result.get("tmdb_found", pd.Series(dtype=bool)).fillna(False).sum()) if not result.empty else 0
    print(f"Done. Found {found}/{len(result)}. Cache: {Path(args.cache).resolve()}")

    if args.discover:
        metadata = metadata_from_cache(None, cache_path=args.cache, include_all=True)
        meta = prepare_metadata(metadata)
        ratings = data["ratings"].copy()
        ratings["Rating"] = pd.to_numeric(ratings.get("Rating"), errors="coerce")
        favorite_ids = set(ratings.loc[ratings["Rating"] >= 4.0, "movie_id"].dropna()) | set(data["likes"].get("movie_id", pd.Series(dtype=str)).dropna())
        favorite_meta = meta[meta["movie_id"].isin(favorite_ids)].copy()
        if "tmdb_popularity" in favorite_meta.columns:
            favorite_meta = favorite_meta.sort_values("tmdb_popularity", ascending=False, na_position="last")
        discovered = discover_movies_from_favorites(favorite_meta, client=client, per_seed=args.per_seed, seed_limit=args.seed_limit, force=args.force)
        print(f"Discovered/cached {len(discovered)} outside-watchlist candidates.")


if __name__ == "__main__":
    main()
