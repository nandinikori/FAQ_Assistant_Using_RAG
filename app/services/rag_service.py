"""RAG orchestration service.

This service combines embedding generation, vector similarity retrieval, and
answer assembly into a single application-facing workflow.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.infrastructure.database import DatabaseClient
from app.infrastructure.embedding_client import GeminiEmbeddingClient
from app.infrastructure.faq_repository import FAQRepository


class RAGService:
    """Coordinates retrieval from the FAQ store and response creation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.database_client = DatabaseClient(settings)
        self.embedding_client = GeminiEmbeddingClient(settings)
        self.repository = FAQRepository(self.database_client)

    def retrieve(self, question: str, limit: int = 5):
        embedding = self.embedding_client.generate_embedding(question)
        return self.repository.search_similar(embedding, limit=limit)

    def answer_question(self, question: str) -> str:
        matches = self.retrieve(question, limit=3)
        if not matches:
            return "I could not find a relevant FAQ in the current knowledge base."

        context = "\n\n".join(
            f"Q: {row['question']}\nA: {row['answer']}" for row in matches
        )
        return (
            "Based on the retrieved FAQs, here is the answer:\n\n"
            f"{context}"
        )
