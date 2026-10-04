"""Vector search helper for nearest-neighbor FAQ retrieval.

This module performs the PostgreSQL pgvector similarity query used by the RAG
pipeline to rank the most relevant FAQ entries for a user query.
"""

from __future__ import annotations

from typing import Sequence

from app.infrastructure.database import DatabaseClient


class FAQRetrieval:
    """Searches the FAQ table for the most similar vector matches."""

    def __init__(self, database_client: DatabaseClient):
        self.database_client = database_client

    def search_similar(self, embedding: Sequence[float], limit: int = 5):
        vector = "[" + ",".join(str(float(value)) for value in embedding) + "]"
        query = """
            SELECT id, question, answer, category,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM faq
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        with self.database_client.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (vector, vector, limit))
                return cur.fetchall()
