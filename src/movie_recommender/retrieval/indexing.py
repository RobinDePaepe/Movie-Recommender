"""Chunk normalized retrieval records and persist local, versioned embeddings."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from movie_recommender.data import movie_database as mdb

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Pin the model snapshot so an unchanged configured identity always means identical weights.
MODEL_REVISION = "680353e66602ffe5bfc6cea86f57a6458ad7e6dd"
CHUNK_ALGORITHM_VERSION = "sentence-boundary-v1"
EMBEDDABLE_CONTENT_TYPES = frozenset({"review_text", "tmdb_overview", "tmdb_genres"})
SPECIAL_TOKEN_RESERVE = 8


class TextEmbedder(Protocol):
    """Backend contract; provider implementations own tokenization and model details."""

    model_id: str
    model_revision: str
    max_input_tokens: int

    def count_tokens(self, text: str) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class LocalSentenceTransformerEmbedder:
    """Lazy local SentenceTransformers backend; review text never leaves the device."""

    def __init__(self, model_id: str = MODEL_ID, revision: str = MODEL_REVISION):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local retrieval embeddings require the optional dependency. "
                "Install it with `pip install -e '.[embeddings]'`."
            ) from exc
        try:
            self._model = SentenceTransformer(model_id, revision=revision)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load local embedding model {model_id}@{revision}: {exc}"
            ) from exc
        self.model_id = model_id
        self.model_revision = revision
        self.max_input_tokens = int(self._model.max_seq_length)
        self._tokenizer = self._model.tokenizer

    def count_tokens(self, text: str) -> int:
        # ``encode`` can emit Transformers' max-sequence warning when counting a full
        # source record before chunking. Tokenization alone reports its full length
        # without applying or warning about the model's truncation limit.
        return len(self._tokenizer.tokenize(text))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    record_id: str
    chunk_index: int
    chunker_version: str
    text: str
    text_hash: str
    metadata_json: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_metadata_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _join_text(parts: Sequence[str]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip()).strip()


def _split_long_unit(text: str, count_tokens, limit: int) -> list[str]:
    """Split one overlong sentence at whitespace, falling back to a character boundary."""
    pieces: list[str] = []
    remainder = text.strip()
    while remainder:
        if count_tokens(remainder) <= limit:
            pieces.append(remainder)
            break

        low, high, best = 1, len(remainder), 0
        while low <= high:
            middle = (low + high) // 2
            if count_tokens(remainder[:middle].strip()) <= limit:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best == 0:
            raise ValueError("The embedder token limit is too small to fit any text.")

        boundary = best
        whitespace = remainder.rfind(" ", 0, best + 1)
        newline = remainder.rfind("\n", 0, best + 1)
        boundary = max(whitespace, newline)
        if boundary <= 0:
            boundary = best
        piece = remainder[:boundary].strip()
        if not piece:
            piece = remainder[:best].strip()
            boundary = best
        pieces.append(piece)
        remainder = remainder[boundary:].lstrip()
    return pieces


def _sentence_units(text: str, count_tokens, limit: int) -> list[str]:
    # Keep punctuation with its sentence; paragraph boundaries are sentence boundaries too.
    units = re.findall(r"[^.!?\n]+(?:[.!?]+(?=\s|$)|\n+|$)", text)
    if not units and text.strip():
        units = [text.strip()]
    result: list[str] = []
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if count_tokens(unit) <= limit:
            result.append(unit)
        else:
            result.extend(_split_long_unit(unit, count_tokens, limit))
    return result


def _tail_under_limit(text: str, count_tokens, limit: int) -> str:
    if limit <= 0 or not text:
        return ""
    if count_tokens(text) <= limit:
        return text
    # Find the earliest suffix that fits, then move to a word boundary where possible.
    low, high, start = 0, len(text), len(text)
    while low <= high:
        middle = (low + high) // 2
        if count_tokens(text[middle:].strip()) <= limit:
            start = middle
            high = middle - 1
        else:
            low = middle + 1
    next_space = re.search(r"\s", text[start:])
    if next_space:
        start += next_space.end()
    return text[start:].strip()


def chunk_text(
    text: str,
    embedder: TextEmbedder,
    *,
    target_tokens: int = 200,
    overlap_tokens: int = 32,
) -> tuple[list[str], str]:
    """Keep short text whole and split long text on sentences with bounded overlap."""
    max_tokens = min(target_tokens, int(embedder.max_input_tokens) - SPECIAL_TOKEN_RESERVE)
    if max_tokens < 1:
        raise ValueError("The embedding model input limit is too small for the safety reserve.")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be nonnegative and smaller than the chunk limit.")
    normalized = (text or "").strip()
    if not normalized:
        return [], f"{CHUNK_ALGORITHM_VERSION}-t{max_tokens}-o{overlap_tokens}"
    count_tokens = embedder.count_tokens
    if count_tokens(normalized) <= max_tokens:
        return [normalized], f"{CHUNK_ALGORITHM_VERSION}-t{max_tokens}-o{overlap_tokens}"

    units = _sentence_units(normalized, count_tokens, max_tokens)
    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = _join_text([*current, unit])
        if not current or count_tokens(candidate) <= max_tokens:
            current.append(unit)
            continue

        previous = _join_text(current)
        chunks.append(previous)
        overlap = _tail_under_limit(previous, count_tokens, overlap_tokens)
        while overlap and count_tokens(_join_text([overlap, unit])) > max_tokens:
            available = max_tokens - count_tokens(unit)
            overlap = _tail_under_limit(overlap, count_tokens, available)
        current = [overlap, unit] if overlap else [unit]

    if current:
        chunks.append(_join_text(current))
    if any(count_tokens(chunk) > max_tokens for chunk in chunks):
        raise ValueError("Chunking produced text beyond the embedder input limit.")
    return chunks, f"{CHUNK_ALGORITHM_VERSION}-t{max_tokens}-o{overlap_tokens}"


def build_chunks(
    records: Iterable[Mapping[str, Any]],
    embedder: TextEmbedder,
    *,
    target_tokens: int = 200,
    overlap_tokens: int = 32,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for record in records:
        content_type = str(record.get("content_type", ""))
        text = str(record.get("text") or "").strip()
        if content_type not in EMBEDDABLE_CONTENT_TYPES or not text:
            continue
        record_id = str(record["record_id"])
        parts, chunker_version = chunk_text(
            text, embedder, target_tokens=target_tokens, overlap_tokens=overlap_tokens
        )
        for chunk_index, chunk_text_value in enumerate(parts):
            chunk_id = _sha256(f"{record_id}\0{chunker_version}\0{chunk_index}")
            metadata = {
                key: _safe_metadata_value(record.get(key))
                for key in (
                    "movie_id", "movie_name", "movie_year", "source", "source_event_id",
                    "content_type", "watched_date", "rating", "rewatch", "tmdb_id",
                    "letterboxd_uri",
                )
            }
            metadata["chunk_index"] = chunk_index
            metadata["chunker_version"] = chunker_version
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                record_id=record_id,
                chunk_index=chunk_index,
                chunker_version=chunker_version,
                text=chunk_text_value,
                text_hash=_sha256(chunk_text_value),
                metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False),
            ))
    return chunks


def _normalized_vectors(raw_vectors: Any, expected_rows: int) -> np.ndarray:
    vectors = np.asarray(raw_vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows or vectors.shape[1] == 0:
        raise ValueError("Embedder returned an invalid vector matrix shape.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedder returned non-finite vector values.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedder returned a zero-length vector.")
    return np.asarray(vectors / norms, dtype=np.float32)


def refresh_retrieval_index(
    db_path: str = str(mdb.DB_PATH),
    embedder: TextEmbedder | None = None,
    *,
    batch_size: int = 32,
    target_tokens: int = 200,
    overlap_tokens: int = 32,
) -> dict[str, Any]:
    """Build current chunks and refresh their active embedding model in SQLite."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    backend = embedder or LocalSentenceTransformerEmbedder()
    mdb.init_db(db_path)
    records = mdb.load_retrieval_corpus(db_path).to_dict(orient="records")
    desired_chunks = build_chunks(
        records, backend, target_tokens=target_tokens, overlap_tokens=overlap_tokens
    )
    model_id = str(backend.model_id)
    model_revision = str(backend.model_revision)
    now = mdb.utc_now()
    desired_ids = {chunk.chunk_id for chunk in desired_chunks}
    embedded_count = 0

    with mdb.connect(db_path) as conn:
        existing_ids = {row[0] for row in conn.execute("SELECT chunk_id FROM retrieval_chunks").fetchall()}
        removed_count = len(existing_ids - desired_ids)
        current_models = conn.execute(
            "SELECT DISTINCT model_id,model_revision FROM retrieval_embeddings"
        ).fetchall()
        model_changed = any(
            row["model_id"] != model_id or row["model_revision"] != model_revision
            for row in current_models
        )

        for chunk in desired_chunks:
            conn.execute(
                """INSERT INTO retrieval_chunks
                   (chunk_id,record_id,chunk_index,chunker_version,text,text_hash,metadata_json,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(chunk_id) DO UPDATE SET
                     record_id=excluded.record_id,chunk_index=excluded.chunk_index,
                     chunker_version=excluded.chunker_version,text=excluded.text,text_hash=excluded.text_hash,
                     metadata_json=excluded.metadata_json,
                     updated_at=CASE WHEN retrieval_chunks.text_hash IS NOT excluded.text_hash
                       OR retrieval_chunks.metadata_json IS NOT excluded.metadata_json
                       THEN excluded.updated_at ELSE retrieval_chunks.updated_at END""",
                (chunk.chunk_id, chunk.record_id, chunk.chunk_index, chunk.chunker_version,
                 chunk.text, chunk.text_hash, chunk.metadata_json, now, now),
            )

        for stale_id in existing_ids - desired_ids:
            conn.execute("DELETE FROM retrieval_chunks WHERE chunk_id=?", (stale_id,))

        if model_changed:
            conn.execute("DELETE FROM retrieval_embeddings")

        pending = conn.execute(
            """SELECT c.chunk_id,c.text,c.text_hash FROM retrieval_chunks c
               LEFT JOIN retrieval_embeddings e ON e.chunk_id=c.chunk_id
               WHERE e.chunk_id IS NULL OR e.model_id<>? OR e.model_revision<>?
                  OR e.source_text_hash<>c.text_hash
               ORDER BY c.chunk_id""",
            (model_id, model_revision),
        ).fetchall()
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            raw = backend.embed([row["text"] for row in batch])
            vectors = _normalized_vectors(raw, len(batch))
            for row, vector in zip(batch, vectors):
                conn.execute(
                    """INSERT INTO retrieval_embeddings
                       (chunk_id,model_id,model_revision,vector_dimension,vector,source_text_hash,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?,?)
                       ON CONFLICT(chunk_id) DO UPDATE SET
                         model_id=excluded.model_id,model_revision=excluded.model_revision,
                         vector_dimension=excluded.vector_dimension,vector=excluded.vector,
                         source_text_hash=excluded.source_text_hash,updated_at=excluded.updated_at""",
                    (row["chunk_id"], model_id, model_revision, int(vector.size),
                     vector.tobytes(order="C"), row["text_hash"], now, now),
                )
            embedded_count += len(batch)

        total_chunks = conn.execute("SELECT COUNT(*) FROM retrieval_chunks").fetchone()[0]
        total_embeddings = conn.execute("SELECT COUNT(*) FROM retrieval_embeddings").fetchone()[0]
        mismatched = conn.execute(
            """SELECT COUNT(*) FROM retrieval_chunks c LEFT JOIN retrieval_embeddings e USING(chunk_id)
               WHERE e.chunk_id IS NULL OR e.source_text_hash<>c.text_hash OR e.model_id<>? OR e.model_revision<>?""",
            (model_id, model_revision),
        ).fetchone()[0]
        if mismatched:
            raise RuntimeError(f"Retrieval index refresh left {mismatched} chunks without current embeddings.")
        dimensions = conn.execute(
            "SELECT DISTINCT vector_dimension FROM retrieval_embeddings"
        ).fetchall()
        if len(dimensions) > 1:
            raise RuntimeError("Retrieval index contains mixed embedding dimensions.")

    return {
        "source_records": len(records),
        "chunks": int(total_chunks),
        "embeddings": int(total_embeddings),
        "embedded": embedded_count,
        "reused": int(total_chunks) - embedded_count,
        "removed": int(removed_count),
        "model_id": model_id,
        "model_revision": model_revision,
    }
