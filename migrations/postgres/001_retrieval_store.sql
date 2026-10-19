CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS retrieval_chunks (
    chunk_id text PRIMARY KEY,
    record_id text NOT NULL,
    movie_id text NOT NULL,
    source text NOT NULL,
    source_event_id text NOT NULL,
    content_type text NOT NULL,
    chunk_index integer NOT NULL,
    chunker_version text NOT NULL,
    text text NOT NULL,
    text_hash text NOT NULL,
    movie_name text,
    movie_year integer,
    watched_date text,
    rating real,
    rewatch boolean,
    tmdb_id bigint,
    letterboxd_uri text,
    model_id text NOT NULL,
    model_revision text NOT NULL,
    embedding vector(384) NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('simple'::regconfig, coalesce(movie_name, '') || ' ' || text)
    ) STORED,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT retrieval_chunks_record_version_index_key
        UNIQUE (record_id, chunker_version, chunk_index)
);

CREATE INDEX IF NOT EXISTS retrieval_chunks_embedding_idx
    ON retrieval_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS retrieval_chunks_search_idx
    ON retrieval_chunks USING gin (search_vector);
