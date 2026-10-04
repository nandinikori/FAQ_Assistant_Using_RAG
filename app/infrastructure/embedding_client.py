"""Gemini embedding client used to vectorize FAQ content.

The embedding layer converts text into numerical vectors that can later be
matched semantically in PostgreSQL using pgvector.
"""

from __future__ import annotations

from typing import Sequence

from app.config.settings import Settings


class GeminiEmbeddingClient:
    """Creates vector embeddings for text using the Gemini embedding model."""

    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")

    def generate_embedding(self, text: str) -> list[float]:
        # Import the SDK lazily to keep the app boot process flexible.
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai"
            ) from exc

        client = genai.Client(api_key=self.settings.gemini_api_key)
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )

        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise RuntimeError("No embedding returned by Gemini.")

        return embeddings[0].values

    def generate_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        # Batch helper for multiple text inputs.
        return [self.generate_embedding(text) for text in texts]
