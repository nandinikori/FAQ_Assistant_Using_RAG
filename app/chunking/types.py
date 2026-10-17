"""Typed source and chunk records with document-level provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    version: str
    source_name: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceBlock:
    block_id: str
    kind: str
    text: str
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    heading_path: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    document_version: str
    source_name: str
    strategy: str
    text: str
    token_count: int
    source_char_start: int
    source_char_end: int
    source_line_start: int
    source_line_end: int
    heading_path: tuple[str, ...]
    source_block_ids: tuple[str, ...]
    parent_chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
