"""Infrastructure adapters for database and embedding integrations.

This layer isolates PostgreSQL and Gemini interaction code from the rest of the
application so services can remain clean and reusable.
"""

from app.infrastructure.database import DatabaseClient
from app.infrastructure.embedding_client import GeminiEmbeddingClient

__all__ = ["DatabaseClient", "GeminiEmbeddingClient"]
