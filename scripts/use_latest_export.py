from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from movie_recommender.data.movie_database import rebuild_database


# Standard Letterboxd export names end with a UTC timestamp, for example:
# letterboxd-bslinky-2026-10-16-18-08-utc.zip
EXPORT_TIMESTAMP = re.compile(r"-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})-utc\.zip$", re.IGNORECASE)


def latest_export(data_dir: Path) -> Path:
    exports: list[tuple[datetime, Path]] = []
    for path in data_dir.glob("*.zip"):
        match = EXPORT_TIMESTAMP.search(path.name)
        if match:
            try:
                timestamp = datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            exports.append((timestamp, path))

    if not exports:
        raise FileNotFoundError(f"No timestamped Letterboxd export ZIPs found in {data_dir}.")
    return max(exports, key=lambda item: (item[0], item[1].name))[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use the newest timestamped Letterboxd export and refresh the local movie database."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing Letterboxd export ZIPs")
    args = parser.parse_args()

    data_dir = args.data_dir
    source = latest_export(data_dir)
    active_export = data_dir / "letterboxd_export.zip"

    if source.resolve() != active_export.resolve():
        shutil.copy2(source, active_export)

    result = rebuild_database(
        export_zip=active_export,
        cache_path=data_dir / "tmdb_cache.json",
        db_path=data_dir / "movie_recommender.sqlite",
    )
    print(f"Using latest Letterboxd export: {source} ({source.stat().st_size:,} bytes)")
    print(f"Updated active export: {active_export}")
    print(f"Rebuilt database: {data_dir / 'movie_recommender.sqlite'}")
    print(result)


if __name__ == "__main__":
    main()
