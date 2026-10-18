CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    document_version TEXT NOT NULL,
    source_name TEXT NOT NULL,
    strategy TEXT NOT NULL,
    chunking_configuration JSONB NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding VECTOR(3072) NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimensions INTEGER NOT NULL,
    embedding_configuration_version TEXT NOT NULL,
    token_count INTEGER NOT NULL CHECK (token_count >= 0),
    source_char_start INTEGER NOT NULL CHECK (source_char_start >= 0),
    source_char_end INTEGER NOT NULL CHECK (source_char_end >= source_char_start),
    source_line_start INTEGER NOT NULL CHECK (source_line_start >= 1),
    source_line_end INTEGER NOT NULL CHECK (source_line_end >= source_line_start),
    heading_path JSONB NOT NULL,
    source_block_ids JSONB NOT NULL,
    parent_chunk_id TEXT,
    metadata JSONB NOT NULL,
    PRIMARY KEY (
        chunk_id,
        embedding_model,
        embedding_dimensions,
        embedding_configuration_version
    )
);
