"""RAG orchestration service.

This service combines embedding generation, vector similarity retrieval, and
answer assembly into a single application-facing workflow.
"""

from __future__ import annotations

from typing import Any

from app.config.settings import Settings
from app.infrastructure.database import DatabaseClient
from app.infrastructure.document_chunk_repository import DocumentChunkRepository
from app.infrastructure.embedding_client import GeminiEmbeddingClient
from app.infrastructure.faq_repository import FAQRepository


class RAGService:
    """Coordinates retrieval from the FAQ store and response creation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.database_client = DatabaseClient(settings)
        self.embedding_client = GeminiEmbeddingClient(settings)
        self.repository = FAQRepository(self.database_client)
        self.document_repository = DocumentChunkRepository(
            self.database_client, settings
        )

    def retrieve(self, question: str, limit: int = 5):
        embedding = self.embedding_client.generate_embedding(question)
        return self.repository.search_similar(embedding, limit=limit)

    def get_context(self, question: str) -> str:
        matches = self.retrieve(question, limit=3)
        if not matches:
            return ""

        context = "\n\n".join(
            f"Q: {row['question']}\nA: {row['answer']}" for row in matches
        )
        return context

    def retrieve_document_chunks(
        self,
        question: str,
        limit: int = 5,
        document_id: str | None = None,
        document_version: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks and their source provenance for a question."""
        embedding = self.embedding_client.generate_embedding(question)
        return self.document_repository.search_similar(
            embedding,
            limit=limit,
            document_id=document_id,
            document_version=document_version,
        )
