"""Run an offline comparison across the implemented chunking strategies."""

from __future__ import annotations

import json
import re

from app.chunking.corpus import load_policy_corpus
from app.chunking.evaluation import evaluate_chunks
from app.chunking.strategies import (
    ParagraphChunker,
    RecursiveChunker,
    SemanticChunker,
    StructureAwareChunker,
    TableAwareChunker,
    TokenWindowChunker,
    parent_child_chunks,
)


def _lexical_similarity(left: str, right: str) -> float:
    """A transparent local proxy; production should inject embedding cosine."""
    words = lambda text: set(re.findall(r"\w+", text.casefold()))
    left_words, right_words = words(left), words(right)
    union = left_words | right_words
    return len(left_words & right_words) / len(union) if union else 1.0


def main() -> None:
    corpus = load_policy_corpus()
    report = []
    strategies = [
        ("token-window", TokenWindowChunker(chunk_size=100, overlap=15), 100),
        ("paragraph", ParagraphChunker(max_tokens=140), 140),
        ("recursive", RecursiveChunker(max_tokens=120), 120),
        ("structure-aware", StructureAwareChunker(max_tokens=160), 160),
        ("table-aware", TableAwareChunker(max_tokens=90), 90),
        (
            "semantic-lexical-proxy",
            SemanticChunker(_lexical_similarity, threshold=0.12, max_tokens=160),
            160,
        ),
        ("parent-child", None, 80),
    ]
    for document in corpus:
        for name, strategy, target in strategies:
            chunks = (
                parent_child_chunks(document, child_size=80, child_overlap=10)
                if strategy is None
                else strategy.chunk(document)
            )
            report.append(
                {
                    "document_id": document.document_id,
                    "version": document.version,
                    "strategy": name,
                    "metrics": evaluate_chunks(document, chunks, target),
                }
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
