"""Repository for persisting and retrieving FAQ records in PostgreSQL.

This layer handles table creation, inserts, and semantic similarity queries.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from app.domain.faq import FAQ
from app.infrastructure.database import DatabaseClient


class FAQRepository:
    """Encapsulates all database operations for the FAQ knowledge base."""

    def __init__(self, database_client: DatabaseClient):
        self.database_client = database_client

    def ensure_schema(self) -> None:
        # Ensure the pgvector extension and FAQ table exist before inserts.
        with self.database_client.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS faq (
                        id SERIAL PRIMARY KEY,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        category TEXT,
                        embedding VECTOR
                    );
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE faq
                    ADD COLUMN IF NOT EXISTS embedding VECTOR;
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE faq
                    ALTER COLUMN embedding TYPE VECTOR;
                    """
                )

    def upsert(self, faq: FAQ, embedding: Sequence[float] | None = None) -> int:
        # Insert a single FAQ record into PostgreSQL, including the vector when present.
        with self.database_client.connect() as conn:
            with conn.cursor() as cur:
                if embedding is None:
                    cur.execute(
                        """
                        INSERT INTO faq (question, answer, category)
                        VALUES (%s, %s, %s)
                        RETURNING id;
                        """,
                        faq.to_insert_tuple(),
                    )
                else:
                    vector = "[" + ",".join(str(float(value)) for value in embedding) + "]"
                    cur.execute(
                        """
                        INSERT INTO faq (question, answer, category, embedding)
                        VALUES (%s, %s, %s, %s::vector)
                        RETURNING id;
                        """,
                        (faq.question, faq.answer, faq.category, vector),
                    )
                row = cur.fetchone()
                return int(row["id"]) if row else 0

    def bulk_upsert(self, faqs: Iterable[FAQ]) -> int:
        # Convenience method for inserting many FAQ rows.
        inserted = 0
        for faq in faqs:
            inserted += 1
            self.upsert(faq)
        return inserted

    def search_similar(self, embedding: Sequence[float], limit: int = 5):
        # Convert embedding list to a PostgreSQL vector literal before querying.
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
