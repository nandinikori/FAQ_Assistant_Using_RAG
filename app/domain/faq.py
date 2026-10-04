"""Domain model for an FAQ entry used by the RAG knowledge base.

This module defines the core FAQ record shape shared by ingestion,
retrieval, and persistence layers.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class FAQ:
    """Represents one question-answer pair in the knowledge base."""

    question: str
    answer: str
    category: Optional[str] = None
    id: Optional[int] = None

    def to_insert_tuple(self) -> tuple[str, str, Optional[str]]:
        """Return the fields needed for database insertion."""
        # Keep the insert payload simple: question, answer, and optional category.
        return self.question, self.answer, self.category
