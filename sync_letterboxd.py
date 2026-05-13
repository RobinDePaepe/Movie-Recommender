from __future__ import annotations

import argparse
from letterboxd_sync import sync_rss, sync_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync recent Letterboxd activity from your public RSS feed.")
    parser.add_argument("username_or_url", help="Letterboxd username, profile URL, or RSS URL, e.g. bslinky or https://letterboxd.com/bslinky/rss/")
    parser.add_argument("--status", action="store_true", help="Print sync status after running.")
    args = parser.parse_args()
    result = sync_rss(args.username_or_url)
    print(f"Fetched {result.get('fetched_events', 0)} RSS events; added {result.get('new_events', 0)} new events; total stored {result.get('total_events', 0)}.")
    if args.status:
        print(sync_status())


if __name__ == "__main__":
    main()
