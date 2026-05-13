from __future__ import annotations

import argparse
from pprint import pprint

from movie_database import DB_PATH, database_status, import_feedback_csv, import_letterboxd_export, import_tmdb_cache, init_db, rebuild_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize or rebuild the SQLite database for the movie recommender.")
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite database path")
    parser.add_argument("--export-zip", default="data/letterboxd_export.zip")
    parser.add_argument("--tmdb-cache", default="data/tmdb_cache.json")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the database from export/cache/feedback")
    parser.add_argument("--status", action="store_true", help="Show database status only")
    args = parser.parse_args()

    if args.status:
        pprint(database_status(args.db))
        return

    if args.rebuild:
        result = rebuild_database(export_zip=args.export_zip, cache_path=args.tmdb_cache, db_path=args.db)
    else:
        init_db(args.db)
        result = {
            "letterboxd": import_letterboxd_export(export_zip=args.export_zip, db_path=args.db),
            "tmdb_metadata": import_tmdb_cache(cache_path=args.tmdb_cache, db_path=args.db),
            "feedback": import_feedback_csv(db_path=args.db),
        }
    pprint(result)
    pprint(database_status(args.db))


if __name__ == "__main__":
    main()
