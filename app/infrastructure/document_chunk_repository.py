"""PostgreSQL persistence and vector retrieval for versioned document chunks."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

from psycopg2.extras import Json

from app.chunking.types import DocumentChunk
from app.config.settings import Settings
from app.infrastructure.database import DatabaseClient


class DocumentChunkRepository:
    """Stores chunks with provenance and an embedding-configuration boundary."""

    def __init__(self, database_client: DatabaseClient, settings: Settings):
        self.database_client = database_client
        self.settings = settings
        if (
            not isinstance(settings.embedding_dimensions, int)
            or isinstance(settings.embedding_dimensions, bool)
            or settings.embedding_dimensions < 1
        ):
            raise ValueError("Embedding dimensions must be a positive integer.")
        if not settings.gemini_embedding_model:
            raise ValueError("An embedding model must be configured.")
        if not settings.embedding_configuration_version:
            raise ValueError("An embedding configuration version must be configured.")

    def ensure_schema(self) -> None:
        """Create the chunk table or reject a schema with incompatible dimensions."""
        dimensions = self.settings.embedding_dimensions
        with self.database_client.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        document_version TEXT NOT NULL,
                        source_name TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        chunking_configuration JSONB NOT NULL,
                        text TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        embedding VECTOR({dimensions}) NOT NULL,
                        embedding_model TEXT NOT NULL,
                        embedding_dimensions INTEGER NOT NULL,
                        embedding_configuration_version TEXT NOT NULL,
                        token_count INTEGER NOT NULL,
                        source_char_start INTEGER NOT NULL,
                        source_char_end INTEGER NOT NULL,
                        source_line_start INTEGER NOT NULL,
                        source_line_end INTEGER NOT NULL,
                        heading_path JSONB NOT NULL,
                        source_block_ids JSONB NOT NULL,
                        parent_chunk_id TEXT,
                        metadata JSONB NOT NULL,
                        PRIMARY KEY (
                            chunk_id,
                            embedding_model,
                            embedding_dimensions,
                            embedding_configuration_version
                        ),
                        CHECK (token_count >= 0),
                        CHECK (source_char_start >= 0),
                        CHECK (source_char_end >= source_char_start),
                        CHECK (source_line_start >= 1),
                        CHECK (source_line_end >= source_line_start)
                    );
                    """
                )
                cur.execute(
                    """
                    SELECT format_type(attribute.atttypid, attribute.atttypmod)
                    FROM pg_attribute AS attribute
                    JOIN pg_class AS relation ON relation.oid = attribute.attrelid
                    WHERE relation.relname = 'document_chunks'
                      AND attribute.attname = 'embedding'
                      AND NOT attribute.attisdropped;
                    """
                )
                row = cur.fetchone()
                expected_type = f"vector({dimensions})"
                actual_type = next(iter(row.values())) if row else None
                if actual_type != expected_type:
                    raise ValueError(
                        "document_chunks.embedding has incompatible dimensions: "
                        f"expected {expected_type}, found {actual_type!r}. "
                        "Use a separate database/schema for this embedding configuration."
                    )

    def existing_chunk_ids(self, chunks: Sequence[DocumentChunk]) -> set[str]:
        """Find chunks already stored for this embedding configuration."""
        if not chunks:
            return set()
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        with self.database_client.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id
                    FROM document_chunks
                    WHERE chunk_id = ANY(%s)
                      AND embedding_model = %s
                      AND embedding_dimensions = %s
                      AND embedding_configuration_version = %s;
                    """,
                    (
                        chunk_ids,
                        self.settings.gemini_embedding_model,
                        self.settings.embedding_dimensions,
                        self.settings.embedding_configuration_version,
                    ),
                )
                return {str(row["chunk_id"]) for row in cur.fetchall()}

    def insert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Insert all provided chunks atomically; existing identities are unchanged."""
        if len(chunks) != len(embeddings):
            raise ValueError("Each document chunk must have exactly one embedding.")

        validated_vectors: list[str] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if len(embedding) != self.settings.embedding_dimensions:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} has embedding dimension {len(embedding)}; "
                    f"expected {self.settings.embedding_dimensions}."
                )
            try:
                values = [float(value) for value in embedding]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} contains a non-numeric embedding value."
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"Chunk {chunk.chunk_id} contains a non-finite embedding value."
                )
            validated_vectors.append(
                "[" + ",".join(str(value) for value in values) + "]"
            )

        inserted = 0
        with self.database_client.connect() as conn:
            with conn.cursor() as cur:
                for chunk, vector in zip(chunks, validated_vectors, strict=True):
                    cur.execute(
                        """
                        INSERT INTO document_chunks (
                            chunk_id, document_id, document_version, source_name,
                            strategy, chunking_configuration, text, content_hash,
                            embedding, embedding_model, embedding_dimensions,
                            embedding_configuration_version, token_count,
                            source_char_start, source_char_end, source_line_start,
                            source_line_end, heading_path, source_block_ids,
                            parent_chunk_id, metadata
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s::vector,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (
                            chunk_id,
                            embedding_model,
                            embedding_dimensions,
                            embedding_configuration_version
                        ) DO NOTHING;
                        """,
                        (
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.document_version,
                            chunk.source_name,
                            chunk.strategy,
                            Json(chunk.metadata),
                            chunk.text,
                            hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                            vector,
                            self.settings.gemini_embedding_model,
                            self.settings.embedding_dimensions,
                            self.settings.embedding_configuration_version,
                            chunk.token_count,
                            chunk.source_char_start,
                            chunk.source_char_end,
                            chunk.source_line_start,
                            chunk.source_line_end,
                            Json(list(chunk.heading_path)),
                            Json(list(chunk.source_block_ids)),
                            chunk.parent_chunk_id,
                            Json(chunk.metadata),
                        ),
                    )
                    inserted += cur.rowcount
        return inserted

    def search_similar(
        self,
        embedding: Sequence[float],
        limit: int = 5,
        document_id: str | None = None,
        document_version: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest chunks, optionally restricted to a document/version."""
        if limit < 1:
            raise ValueError("limit must be positive.")
        if len(embedding) != self.settings.embedding_dimensions:
            raise ValueError(
                f"Query embedding dimension {len(embedding)} does not match "
                f"configured dimension {self.settings.embedding_dimensions}."
            )
        vector = "[" + ",".join(str(float(value)) for value in embedding) + "]"
        filters = [
            "embedding_model = %s",
            "embedding_dimensions = %s",
            "embedding_configuration_version = %s",
        ]
        params: list[Any] = [
            self.settings.gemini_embedding_model,
            self.settings.embedding_dimensions,
            self.settings.embedding_configuration_version,
        ]
        if document_id is not None:
            filters.append("document_id = %s")
            params.append(document_id)
        if document_version is not None:
            filters.append("document_version = %s")
            params.append(document_version)
        query_params = [vector, *params, vector, limit]
        query = f"""
            SELECT chunk_id, document_id, document_version, source_name,
                   strategy, chunking_configuration, text, content_hash,
                   token_count, source_char_start, source_char_end,
                   source_line_start, source_line_end, heading_path,
                   source_block_ids, parent_chunk_id, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            WHERE {" AND ".join(filters)}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        with self.database_client.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(query_params))
                return cur.fetchall()
