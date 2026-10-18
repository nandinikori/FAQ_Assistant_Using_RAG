"""Idempotent document chunk ingestion into PostgreSQL + pgvector."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Protocol

from app.chunking.corpus import load_policy_corpus
from app.chunking.strategies import (
    ParagraphChunker,
    RecursiveChunker,
    StructureAwareChunker,
    TableAwareChunker,
)
from app.chunking.types import Document, DocumentChunk
from app.config.settings import Settings, get_settings
from app.infrastructure.database import DatabaseClient
from app.infrastructure.document_chunk_repository import DocumentChunkRepository
from app.infrastructure.embedding_client import GeminiEmbeddingClient


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[DocumentChunk]: ...


@dataclass(frozen=True, slots=True)
class DocumentIngestionResult:
    document_id: str
    document_version: str
    chunk_ids: tuple[str, ...]
    inserted_count: int
    already_present_count: int


class ChunkIngestionError(RuntimeError):
    """Raised when any chunk fails embedding, before this batch is written."""

    def __init__(self, failures: list[tuple[str, str]]):
        self.failures = tuple(failures)
        details = "; ".join(
            f"{chunk_id}: {message}" for chunk_id, message in self.failures
        )
        super().__init__(f"Document ingestion aborted; failed chunks: {details}")


class DocumentIngestionService:
    """Chunks, embeds, and atomically persists a versioned source document."""

    def __init__(
        self,
        repository: DocumentChunkRepository,
        embedding_client: GeminiEmbeddingClient,
    ):
        settings = repository.settings
        if (
            embedding_client.model != settings.gemini_embedding_model
            or embedding_client.dimensions != settings.embedding_dimensions
            or embedding_client.configuration_version
            != settings.embedding_configuration_version
        ):
            raise ValueError(
                "Embedding client and document repository configurations differ."
            )
        self.repository = repository
        self.embedding_client = embedding_client

    def ingest(
        self,
        document: Document,
        chunker: Chunker,
    ) -> DocumentIngestionResult:
        chunks = chunker.chunk(document)
        self.repository.ensure_schema()
        existing_ids = self.repository.existing_chunk_ids(chunks)
        missing_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]

        embeddings: list[list[float]] = []
        failures: list[tuple[str, str]] = []
        for chunk in missing_chunks:
            try:
                embedding = self.embedding_client.generate_embedding(chunk.text)
                if len(embedding) != self.embedding_client.dimensions:
                    raise ValueError(
                        f"embedding has dimension {len(embedding)}; "
                        f"expected {self.embedding_client.dimensions}"
                    )
                embeddings.append(embedding)
            except Exception as exc:
                failures.append((chunk.chunk_id, str(exc)))
        if failures:
            raise ChunkIngestionError(failures)

        inserted_count = self.repository.insert_chunks(missing_chunks, embeddings)
        return DocumentIngestionResult(
            document_id=document.document_id,
            document_version=document.version,
            chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
            inserted_count=inserted_count,
            already_present_count=len(chunks) - inserted_count,
        )


def _build_chunker(strategy: str, max_tokens: int) -> Chunker:
    chunkers: dict[str, type[ParagraphChunker] | type[RecursiveChunker]
                   | type[StructureAwareChunker] | type[TableAwareChunker]] = {
        "paragraph": ParagraphChunker,
        "recursive": RecursiveChunker,
        "structure-aware": StructureAwareChunker,
        "table-aware": TableAwareChunker,
    }
    try:
        return chunkers[strategy](max_tokens=max_tokens)
    except KeyError as exc:
        raise ValueError(f"Unsupported chunking strategy: {strategy}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a versioned policy document.")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--strategy",
        choices=("paragraph", "recursive", "structure-aware", "table-aware"),
        default="structure-aware",
    )
    parser.add_argument("--max-tokens", type=int, default=180)
    arguments = parser.parse_args()

    matches = [
        document
        for document in load_policy_corpus()
        if document.document_id == arguments.document_id
        and document.version == arguments.version
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one corpus document for {arguments.document_id!r} "
            f"version {arguments.version!r}; found {len(matches)}."
        )

    settings: Settings = get_settings()
    database_client = DatabaseClient(settings)
    embedding_client = GeminiEmbeddingClient(settings)
    repository = DocumentChunkRepository(database_client, settings)
    result = DocumentIngestionService(repository, embedding_client).ingest(
        matches[0], _build_chunker(arguments.strategy, arguments.max_tokens)
    )
    print(
        f"Ingested {result.inserted_count} new chunks for "
        f"{result.document_id} v{result.document_version}; "
        f"{result.already_present_count} already existed."
    )


if __name__ == "__main__":
    main()
