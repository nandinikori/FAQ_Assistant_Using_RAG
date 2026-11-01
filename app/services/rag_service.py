"""RAG orchestration service.

This service combines embedding generation, vector similarity retrieval, and
answer assembly into a single application-facing workflow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.config.settings import Settings
from app.infrastructure.database import DatabaseClient
from app.infrastructure.document_chunk_repository import DocumentChunkRepository
from app.infrastructure.embedding_client import GeminiEmbeddingClient
from app.infrastructure.faq_repository import FAQRepository
from app.retrieval.retrieval import DocumentRetriever, RetrievalMethod


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
        self.document_retriever = DocumentRetriever(
            self.document_repository, self.embedding_client
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
        *,
        method: RetrievalMethod = "dense",
        strategy: str | None = None,
        candidate_limit: int = 1000,
        chunking_configuration: Mapping[str, Any] | None = None,
        document_ids: Sequence[str] | None = None,
        document_versions: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve document chunks with dense, BM25, RRF, or reranked search."""
        if method not in ("dense", "bm25", "rrf", "rerank"):
            raise ValueError(f"Unsupported retrieval method: {method}")
        retriever = self.document_retriever
        embedding = (
            None
            if method == "bm25"
            else self.embedding_client.generate_embedding(question)
        )
        return retriever.retrieve(
            question,
            method=method,
            limit=limit,
            embedding=embedding,
            document_id=document_id,
            document_version=document_version,
            strategy=strategy,
            candidate_limit=candidate_limit,
            chunking_configuration=chunking_configuration,
            document_ids=document_ids,
            document_versions=document_versions,
        )
