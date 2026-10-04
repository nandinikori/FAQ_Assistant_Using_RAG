"""Standalone FAQ data loader for ingestion workflows.

This module converts raw source data into the domain model used by the RAG
system before it is embedded and stored.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.domain.faq import FAQ


class FAQDataIngestion:
    """Reads FAQ source data and converts each row into a domain object."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def load_faqs(self) -> list[FAQ]:
        with self.file_path.open("r", encoding="utf-8") as source:
            payload = json.load(source)

        if isinstance(payload, dict):
            records = payload.get("faqs", [])
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError(f"Unsupported FAQ payload: {type(payload).__name__}")

        return [FAQ(**record) for record in records]
