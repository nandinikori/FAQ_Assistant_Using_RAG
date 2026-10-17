"""Offline structural metrics for comparing chunking strategies."""

from __future__ import annotations

from collections.abc import Sequence

from app.chunking.types import Document, DocumentChunk


def evaluate_chunks(
    document: Document,
    chunks: Sequence[DocumentChunk],
    target_max_tokens: int,
) -> dict[str, int | float]:
    """Summarize size, source coverage, and limit violations without relevance claims."""
    intervals = sorted(
        (
            max(0, chunk.source_char_start),
            min(len(document.content), chunk.source_char_end),
        )
        for chunk in chunks
        if chunk.source_char_end > chunk.source_char_start
    )
    covered = 0
    current_start = current_end = None
    for start, end in intervals:
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end
    if current_start is not None and current_end is not None:
        covered += current_end - current_start

    token_counts = [chunk.token_count for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "average_token_count": (
            round(sum(token_counts) / len(token_counts), 2) if token_counts else 0.0
        ),
        "maximum_token_count": max(token_counts, default=0),
        "chunks_over_target": sum(count > target_max_tokens for count in token_counts),
        "source_coverage_percent": (
            round(100 * covered / len(document.content), 2)
            if document.content
            else 100.0
        ),
    }
