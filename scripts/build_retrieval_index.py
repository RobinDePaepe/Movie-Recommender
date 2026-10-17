from __future__ import annotations

import argparse
from pprint import pprint

from movie_recommender.data.movie_database import DB_PATH
from movie_recommender.retrieval.indexing import refresh_retrieval_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build provenance-aware retrieval chunks and local embeddings in SQLite."
    )
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite database path")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--max-chunk-tokens", type=int, default=200)
    parser.add_argument("--overlap-tokens", type=int, default=32)
    args = parser.parse_args()
    result = refresh_retrieval_index(
        db_path=args.db,
        batch_size=args.batch_size,
        target_tokens=args.max_chunk_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    pprint(result)


if __name__ == "__main__":
    main()
