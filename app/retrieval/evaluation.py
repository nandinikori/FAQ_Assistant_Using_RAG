"""Reproducible, embedding-based evaluation of document chunking strategies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.chunking.corpus import load_policy_corpus
from app.chunking.strategies import (
    ParagraphChunker,
    RecursiveChunker,
    SemanticChunker,
    StructureAwareChunker,
    TableAwareChunker,
    TokenWindowChunker,
    parent_child_chunks,
)
from app.chunking.types import Document, DocumentChunk
from app.retrieval.retrieval import (
    EmbeddingReranker,
    bm25_rank,
    candidate_union,
    cosine_similarity,
    reciprocal_rank_fusion,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = PROJECT_ROOT / "data" / "policies" / "retrieval_eval_queries.json"
TARGET_TOKENS = 80
PARENT_MAX_TOKENS = 160
CHILD_OVERLAP = 10
RRF_CONSTANT = 60
RERANK_CANDIDATE_K = 20


class EmbeddingProvider(Protocol):
    """Minimal interface needed by the local ranking benchmark."""

    model: str
    dimensions: int
    configuration_version: str

    def generate_embedding(self, text: str) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    question_id: str
    question: str
    document_id: str
    document_version: str
    required_evidence: tuple[str, ...]


class _EmbeddingCache:
    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider
        self.vectors: dict[str, tuple[float, ...]] = {}
        self.dimensions = provider.dimensions

    def embed(self, text: str) -> tuple[float, ...]:
        if text not in self.vectors:
            vector = tuple(
                float(value) for value in self.provider.generate_embedding(text)
            )
            if not vector:
                raise ValueError("Embedding provider returned an empty vector.")
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Embedding has dimension {len(vector)}; expected {self.dimensions}."
                )
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("Embedding provider returned a non-finite value.")
            self.vectors[text] = vector
        return self.vectors[text]


class _CachedEmbeddingProvider:
    def __init__(self, provider: EmbeddingProvider, cache: _EmbeddingCache):
        self.model = provider.model
        self.dimensions = provider.dimensions
        self.configuration_version = provider.configuration_version
        self.cache = cache

    def generate_embedding(self, text: str) -> Sequence[float]:
        return self.cache.embed(text)


def load_evaluation_questions(
    path: str | Path = DEFAULT_QUESTIONS,
) -> tuple[EvaluationQuestion, ...]:
    """Load and validate fixed source-evidence labels from JSON."""
    with Path(path).open("r", encoding="utf-8") as query_file:
        payload = json.load(query_file)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Evaluation query file must contain a non-empty JSON list.")

    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(payload):
        if not isinstance(record, Mapping):
            raise ValueError(f"Evaluation query {index} must be a JSON object.")
        question_id = record.get("id")
        question = record.get("question")
        document_id = record.get("document_id")
        document_version = record.get("document_version")
        evidence = record.get("required_evidence")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (question_id, question, document_id, document_version)
        ):
            raise ValueError(
                f"Evaluation query {index} requires non-empty id, question, "
                "document_id, and document_version strings."
            )
        if question_id in seen_ids:
            raise ValueError(f"Duplicate evaluation query id: {question_id}")
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(item, str) or not item.strip() for item in evidence)
        ):
            raise ValueError(
                f"Evaluation query {question_id} requires non-empty evidence strings."
            )
        seen_ids.add(question_id)
        questions.append(
            EvaluationQuestion(
                question_id=question_id,
                question=question,
                document_id=document_id,
                document_version=document_version,
                required_evidence=tuple(evidence),
            )
        )
    return tuple(questions)


def run_retrieval_evaluation(
    documents: Sequence[Document],
    questions: Sequence[EvaluationQuestion],
    embedding_provider: EmbeddingProvider,
    *,
    top_k: int = 3,
) -> dict:
    """Compare every existing strategy against fixed, labeled evidence queries."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if not documents or not questions:
        raise ValueError("At least one document and one evaluation question are required.")

    documents_by_key = {
        (document.document_id, document.version): document for document in documents
    }
    if len(documents_by_key) != len(documents):
        raise ValueError("Document IDs and versions must be unique in the corpus.")
    question_ids = [question.question_id for question in questions]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Evaluation question IDs must be unique.")
    for question in questions:
        if not question.required_evidence or len(set(question.required_evidence)) != len(
            question.required_evidence
        ):
            raise ValueError(
                f"Query {question.question_id} must have unique required evidence."
            )
        key = (question.document_id, question.document_version)
        if key not in documents_by_key:
            raise ValueError(
                f"Query {question.question_id} references missing document "
                f"{question.document_id} version {question.document_version}."
            )
        source = _normalize_text(documents_by_key[key].content)
        for evidence in question.required_evidence:
            if _normalize_text(evidence) not in source:
                raise ValueError(
                    f"Query {question.question_id} evidence is absent from its source: "
                    f"{evidence!r}."
                )

    cache = _EmbeddingCache(embedding_provider)
    candidate_sets: dict[str, dict[tuple[str, str], list[DocumentChunk]]] = {}
    strategy_configs = _chunk_corpus(documents, cache)
    for strategy_name, config, chunks_by_key in strategy_configs:
        candidate_sets[strategy_name] = chunks_by_key

    strategy_reports = []
    combination_reports = []
    for strategy_name, config, _ in strategy_configs:
        candidates_by_key = candidate_sets[strategy_name]
        methodology_stats = {
            name: {"queries": [], "hits": [], "recalls": [], "reciprocal_ranks": []}
            for name in ("dense", "bm25", "rrf", "dense-rerank")
        }
        for question in questions:
            key = (question.document_id, question.document_version)
            strategy_chunks = candidates_by_key.get(key, [])
            if strategy_name == "parent-child":
                candidates = [
                    chunk
                    for chunk in strategy_chunks
                    if chunk.metadata.get("role") == "child"
                ]
                parents = {
                    chunk.chunk_id: chunk
                    for chunk in strategy_chunks
                    if chunk.metadata.get("role") == "parent"
                }
            else:
                candidates = strategy_chunks
                parents = {}
            query_vector = cache.embed(question.question)
            dense_ranked = sorted(
                (
                    (
                        cosine_similarity(query_vector, cache.embed(chunk.text)),
                        chunk,
                    )
                    for chunk in candidates
                ),
                key=lambda result: (-result[0], result[1].chunk_id),
            )
            row_by_id = {chunk.chunk_id: _chunk_row(chunk) for chunk in candidates}
            bm25_rows = bm25_rank(
                question.question,
                list(row_by_id.values()),
                limit=max(1, len(row_by_id)),
            )
            bm25_ranked = _rows_to_ranked_chunks(bm25_rows)
            rrf_rows = reciprocal_rank_fusion(
                _ranked_chunks_to_rows(dense_ranked),
                bm25_rows,
                limit=max(1, len(candidates)),
                rrf_constant=RRF_CONSTANT,
            )
            rrf_ranked = _rows_to_ranked_chunks(rrf_rows)
            rerank_rows = candidate_union(
                _ranked_chunks_to_rows(dense_ranked),
                bm25_rows,
                candidate_limit=RERANK_CANDIDATE_K,
            )
            reranked = _rows_to_ranked_chunks(
                EmbeddingReranker(
                    _CachedEmbeddingProvider(embedding_provider, cache)
                ).rerank(
                    question.question, rerank_rows, limit=max(1, len(rerank_rows))
                )
            )

            rankings = {
                "dense": dense_ranked,
                "bm25": bm25_ranked,
                "rrf": rrf_ranked,
                "dense-rerank": reranked,
            }
            for method_name, ranking in rankings.items():
                query_report, hit, recall, reciprocal_rank = _evaluate_ranking(
                    question=question,
                    ranked=ranking,
                    parents=parents,
                    top_k=top_k,
                )
                stats = methodology_stats[method_name]
                stats["queries"].append(query_report)
                stats["hits"].append(hit)
                stats["recalls"].append(recall)
                stats["reciprocal_ranks"].append(reciprocal_rank)

        methodologies = {}
        for method_name, stats in methodology_stats.items():
            metrics = {
                "hit_at_k": round(_mean(stats["hits"]), 4),
                "evidence_recall_at_k": round(_mean(stats["recalls"]), 4),
                "mrr_at_k": round(_mean(stats["reciprocal_ranks"]), 4),
            }
            methodologies[method_name] = {
                "metrics": metrics,
                "queries": stats["queries"],
            }
            combination_reports.append(
                {
                    "strategy": strategy_name,
                    "methodology": method_name,
                    **metrics,
                    "retrieval_chunk_count": sum(
                        sum(
                            chunk.metadata.get("role") == "child"
                            for chunk in chunks
                        )
                        if strategy_name == "parent-child"
                        else len(chunks)
                        for chunks in candidates_by_key.values()
                    ),
                }
            )

        strategy_reports.append({
            "strategy": strategy_name,
            "configuration": config,
            "retrieval_chunk_count": sum(
                sum(chunk.metadata.get("role") == "child" for chunk in chunks)
                if strategy_name == "parent-child"
                else len(chunks)
                for chunks in candidates_by_key.values()
            ),
            "metrics": methodologies["dense"]["metrics"],
            "queries": methodologies["dense"]["queries"],
            "methodologies": methodologies,
        })

    combination_reports.sort(
        key=lambda item: (
            -item["hit_at_k"],
            -item["evidence_recall_at_k"],
            -item["mrr_at_k"],
            item["retrieval_chunk_count"],
            item["strategy"],
            item["methodology"],
        )
    )
    ranked_strategies = sorted(
        strategy_reports,
        key=lambda item: (
            -item["metrics"]["hit_at_k"],
            -item["metrics"]["evidence_recall_at_k"],
            -item["metrics"]["mrr_at_k"],
            item["retrieval_chunk_count"],
            item["strategy"],
        ),
    )
    corpus_fingerprint = _fingerprint(
        [
            {
                "document_id": document.document_id,
                "version": document.version,
                "source_name": document.source_name,
                "content": document.content,
                "metadata": document.metadata,
            }
            for document in sorted(
                documents, key=lambda item: (item.document_id, item.version)
            )
        ]
    )
    query_fingerprint = _fingerprint(
        [
            {
                "id": question.question_id,
                "question": question.question,
                "document_id": question.document_id,
                "document_version": question.document_version,
                "required_evidence": question.required_evidence,
            }
            for question in questions
        ]
    )
    return {
        "schema_version": 1,
        "top_k": top_k,
        "embedding": {
            "model": embedding_provider.model,
            "dimensions": embedding_provider.dimensions,
            "configuration_version": embedding_provider.configuration_version,
            "vectors_fingerprint_sha256": _fingerprint(
                [
                    {"text": text, "vector": vector}
                    for text, vector in sorted(cache.vectors.items())
                ]
            ),
        },
        "target_tokens": TARGET_TOKENS,
        "parent_max_tokens": PARENT_MAX_TOKENS,
        "child_overlap": CHILD_OVERLAP,
        "rrf_constant": RRF_CONSTANT,
        "rerank_candidate_k": RERANK_CANDIDATE_K,
        "corpus_fingerprint_sha256": corpus_fingerprint,
        "query_fingerprint_sha256": query_fingerprint,
        "question_count": len(questions),
        "unique_embedding_text_count": len(cache.vectors),
        "embedding_call_count": len(cache.vectors),
        "ranked_strategies": [
            {
                "strategy": item["strategy"],
                **item["metrics"],
                "retrieval_chunk_count": item["retrieval_chunk_count"],
            }
            for item in ranked_strategies
        ],
        "winner": (
            f"{combination_reports[0]['strategy']} + "
            f"{combination_reports[0]['methodology']}"
        ),
        "winner_strategy": combination_reports[0]["strategy"],
        "winner_methodology": combination_reports[0]["methodology"],
        "ranked_combinations": combination_reports,
        "strategies": strategy_reports,
    }


def _chunk_row(chunk: DocumentChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "heading_path": list(chunk.heading_path),
        "source_char_start": chunk.source_char_start,
        "source_char_end": chunk.source_char_end,
        "_chunk": chunk,
    }


def _ranked_chunks_to_rows(
    ranking: Sequence[tuple[float, DocumentChunk]],
) -> list[dict[str, Any]]:
    rows = []
    for score, chunk in ranking:
        row = _chunk_row(chunk)
        row["retrieval_score"] = score
        rows.append(row)
    return rows


def _rows_to_ranked_chunks(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[float, DocumentChunk]]:
    return [
        (float(row["retrieval_score"]), row["_chunk"])
        for row in rows
    ]


def _evaluate_ranking(
    *,
    question: EvaluationQuestion,
    ranked: Sequence[tuple[float, DocumentChunk]],
    parents: Mapping[str, DocumentChunk],
    top_k: int,
) -> tuple[dict, float, float, float]:
    evidence_ranks: dict[str, int | None] = {}
    for evidence in question.required_evidence:
        evidence_rank = None
        for rank, (_, chunk) in enumerate(ranked, start=1):
            context_text = chunk.text
            if chunk.parent_chunk_id:
                parent = parents.get(chunk.parent_chunk_id)
                if parent is None:
                    raise ValueError(
                        f"Parent chunk {chunk.parent_chunk_id} is missing "
                        f"for child {chunk.chunk_id}."
                    )
                context_text = f"{parent.text}\n{context_text}"
            if _contains_evidence(context_text, evidence):
                evidence_rank = rank
                break
        evidence_ranks[evidence] = evidence_rank

    retrieved = [
        rank is not None and rank <= top_k for rank in evidence_ranks.values()
    ]
    hit = float(all(retrieved))
    recall = sum(retrieved) / len(retrieved)
    first_relevant = min(
        (rank for rank in evidence_ranks.values() if rank is not None),
        default=None,
    )
    reciprocal_rank = (
        1 / first_relevant
        if first_relevant is not None and first_relevant <= top_k
        else 0.0
    )
    report = {
        "query_id": question.question_id,
        "evidence_ranks": evidence_ranks,
        "hit_at_k": int(all(retrieved)),
        "evidence_recall_at_k": round(recall, 4),
        "top_chunks": [
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "score": round(score, 8),
                "source_char_start": chunk.source_char_start,
                "source_char_end": chunk.source_char_end,
            }
            for rank, (score, chunk) in enumerate(ranked[:top_k], start=1)
        ],
    }
    return report, hit, recall, reciprocal_rank


def _chunk_corpus(
    documents: Sequence[Document],
    cache: _EmbeddingCache,
) -> list[tuple[str, dict, dict[tuple[str, str], list[DocumentChunk]]]]:
    def semantic_similarity(left: str, right: str) -> float:
        return cosine_similarity(cache.embed(left), cache.embed(right))

    configurations = (
        (
            "token-window",
            {"chunk_size": TARGET_TOKENS, "overlap": 12},
            lambda document: TokenWindowChunker(
                chunk_size=TARGET_TOKENS, overlap=12
            ).chunk(document),
        ),
        (
            "paragraph",
            {"max_tokens": TARGET_TOKENS},
            lambda document: ParagraphChunker(max_tokens=TARGET_TOKENS).chunk(document),
        ),
        (
            "recursive",
            {"max_tokens": TARGET_TOKENS},
            lambda document: RecursiveChunker(max_tokens=TARGET_TOKENS).chunk(document),
        ),
        (
            "structure-aware",
            {"max_tokens": TARGET_TOKENS},
            lambda document: StructureAwareChunker(max_tokens=TARGET_TOKENS).chunk(document),
        ),
        (
            "table-aware",
            {"max_tokens": TARGET_TOKENS},
            lambda document: TableAwareChunker(max_tokens=TARGET_TOKENS).chunk(document),
        ),
        (
            "semantic",
            {"max_tokens": TARGET_TOKENS, "threshold": 0.35},
            lambda document: SemanticChunker(
                semantic_similarity, threshold=0.35, max_tokens=TARGET_TOKENS
            ).chunk(document),
        ),
        (
            "parent-child",
            {
                "parent_max_tokens": PARENT_MAX_TOKENS,
                "child_size": TARGET_TOKENS,
                "child_overlap": CHILD_OVERLAP,
            },
            lambda document: parent_child_chunks(
                document,
                parent_max_tokens=PARENT_MAX_TOKENS,
                child_size=TARGET_TOKENS,
                child_overlap=CHILD_OVERLAP,
            ),
        ),
    )
    result = []
    for name, config, chunk_document in configurations:
        chunks_by_key = {}
        for document in documents:
            chunks = chunk_document(document)
            chunks_by_key[(document.document_id, document.version)] = chunks
        result.append((name, config, chunks_by_key))
    return result


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold().strip()


def _contains_evidence(text: str, evidence: str) -> bool:
    return _normalize_text(evidence) in _normalize_text(text)


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare chunking strategies against labeled policy evidence."
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the full JSON report (printed to stdout regardless).",
    )
    arguments = parser.parse_args()

    from app.config.settings import get_settings
    from app.infrastructure.embedding_client import GeminiEmbeddingClient

    embedding_client = GeminiEmbeddingClient(get_settings())
    report = run_retrieval_evaluation(
        load_policy_corpus(),
        load_evaluation_questions(arguments.queries),
        embedding_client,
        top_k=arguments.top_k,
    )
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if arguments.output is not None:
        arguments.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
