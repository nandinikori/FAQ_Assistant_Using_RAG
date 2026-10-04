"""FAQ ingestion service.

This module reads FAQ records from a JSON file, prepares them for storage, and
creates vector embeddings before persisting the data in PostgreSQL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from app.domain.faq import FAQ
from app.infrastructure.database import DatabaseClient
from app.infrastructure.embedding_client import GeminiEmbeddingClient
from app.infrastructure.faq_repository import FAQRepository


class FAQIngestionService:
    """Handles loading and inserting FAQ records into the knowledge base."""

    def __init__(self, database_client: DatabaseClient, embedding_client: GeminiEmbeddingClient):
        self.database_client = database_client
        self.embedding_client = embedding_client
        self.repository = FAQRepository(database_client)

    def load_json(self, file_path: str | Path) -> list[FAQ]:
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source)

        if isinstance(payload, dict):
            records = payload.get("faqs", [])
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError(f"Unsupported FAQ payload type: {type(payload).__name__}")

        return [FAQ(**record) for record in records]

    def ingest(self, faqs: Iterable[FAQ]) -> int:
        self.repository.ensure_schema()
        inserted = 0
        for faq in faqs:
            combined_text = f"Question: {faq.question}\nAnswer: {faq.answer}"
            try:
                embedding = self.embedding_client.generate_embedding(combined_text)
            except Exception as exc:  # pragma: no cover
                print(f"Skipping embedding for '{faq.question}' due to error: {exc}")
                continue

            self.repository.upsert(faq, embedding)
            inserted += 1
        return inserted

    def ingest_from_file(self, file_path: str | Path) -> int:
        faqs = self.load_json(file_path)
        return self.ingest(faqs)
