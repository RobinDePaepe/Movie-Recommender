from __future__ import annotations

import argparse
from pprint import pprint

from dotenv import load_dotenv

from movie_recommender.data.movie_database import DB_PATH
from movie_recommender.retrieval.publishing import publish_retrieval_index


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Publish the active SQLite retrieval chunks and embeddings to PostgreSQL/pgvector."
    )
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite database path")
    parser.add_argument("--batch-size", type=int, default=500, help="Rows written per batch")
    args = parser.parse_args()
    pprint(publish_retrieval_index(db_path=args.db, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
