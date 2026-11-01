"""Reproducible, source-coordinate-aware retrieval validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.chunking.configuration import chunking_configuration_fingerprint
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
from app.config.settings import get_settings
from app.infrastructure.embedding_client import GeminiEmbeddingClient
from app.retrieval.retrieval import (
    EmbeddingReranker,
    bm25_rank,
    candidate_union,
    cosine_similarity,
    reciprocal_rank_fusion,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "policies" / "retrieval_eval_queries.json"
DEFAULT_TARGET_TOKENS = 80
DEFAULT_CANDIDATE_LIMIT = 1000
DEFAULT_CONTEXT_BUDGET_TOKENS = 320
DEFAULT_SUCCESS_K = 5
EVIDENCE_SPAN_COVERAGE = 0.5
INITIAL_METHODS = ("dense", "bm25", "rrf", "rerank")


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int
    configuration_version: str

    def generate_embedding(self, text: str) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    evidence_id: str
    document_id: str
    document_version: str
    source_path: str
    line_start: int
    line_end: int
    source_char_start: int
    source_char_end: int
    text: str


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    question_id: str
    group_id: str
    split: str
    category: str
    question: str
    answerability: str
    reference_answer: str | None
    expected_refusal: str | None
    retrieval_scope: Mapping[str, Any]
    evidence: tuple[EvidenceUnit, ...]
    review_status: str


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    questions: tuple[EvaluationQuestion, ...]
    split_manifest: Mapping[str, Any]
    sha256: str


class _EmbeddingCache:
    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider
        self.vectors: dict[str, tuple[float, ...]] = {}
        self.calls = 0
        self.cache_hits = 0
        self.call_seconds = 0.0
        self.cache_lookup_seconds = 0.0
        self.usage: dict[str, int] = {}
        self.cold_calls = 0
        self.warm_hits = 0

    def embed(self, text: str) -> tuple[float, ...]:
        started = time.perf_counter()
        if text in self.vectors:
            self.cache_lookup_seconds += time.perf_counter() - started
            self.cache_hits += 1
            self.warm_hits += 1
            return self.vectors[text]
        vector = tuple(float(value) for value in self.provider.generate_embedding(text))
        self.call_seconds += time.perf_counter() - started
        if not vector:
            raise ValueError("Embedding provider returned an empty vector.")
        if len(vector) != self.provider.dimensions:
            raise ValueError(
                f"Embedding has dimension {len(vector)}; "
                f"expected {self.provider.dimensions}."
            )
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding provider returned a non-finite value.")
        self.vectors[text] = vector
        self.calls += 1
        self.cold_calls += 1
        provider_usage = getattr(self.provider, "last_usage_metadata", None)
        if isinstance(provider_usage, Mapping):
            for name, value in provider_usage.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    self.usage[str(name)] = self.usage.get(str(name), 0) + value
        return vector


class _CachedProvider:
    def __init__(self, cache: _EmbeddingCache):
        self.cache = cache

    def generate_embedding(self, text: str) -> Sequence[float]:
        return self.cache.embed(text)


def load_evaluation_dataset(
    path: str | Path = DEFAULT_DATASET,
    documents: Sequence[Document] | None = None,
) -> EvaluationDataset:
    """Load grouped labels and validate every positive evidence span in source."""
    dataset_path = Path(path)
    raw_bytes = dataset_path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 2:
        raise ValueError("Evaluation dataset must use schema_version 2.")
    if payload.get("coordinate_convention") != (
        "One-based inclusive Markdown source lines; evidence text must occur "
        "verbatim inside the indicated lines. Runtime derives zero-based "
        "half-open character offsets from the exact text."
    ):
        raise ValueError("Unsupported evaluation source-coordinate convention.")
    groups = payload.get("groups")
    manifest = payload.get("split_manifest")
    if not isinstance(groups, list) or not groups or not isinstance(manifest, Mapping):
        raise ValueError("Evaluation dataset requires groups and a split manifest.")

    corpus = list(load_policy_corpus() if documents is None else documents)
    documents_by_key = {
        (document.document_id, document.version): document for document in corpus
    }
    if len(documents_by_key) != len(corpus):
        raise ValueError("Corpus document ID/version pairs must be unique.")

    dev_groups = manifest.get("dev_groups")
    held_out_groups = manifest.get("held_out_groups")
    if (
        not isinstance(dev_groups, list)
        or not isinstance(held_out_groups, list)
        or set(dev_groups).intersection(held_out_groups)
    ):
        raise ValueError("Development and held-out group lists must be disjoint.")
    expected_splits = {
        **{group_id: "dev" for group_id in dev_groups},
        **{group_id: "held_out" for group_id in held_out_groups},
    }

    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    for group_index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ValueError(f"Dataset group {group_index} must be an object.")
        group_id = group.get("group_id")
        split = group.get("split")
        category = group.get("category")
        answerability = group.get("answerability")
        status = group.get("review_status")
        variants = group.get("variants")
        evidence_payload = group.get("gold_evidence")
        scope = group.get("retrieval_scope")
        if not all(isinstance(value, str) and value for value in (
            group_id, split, category, answerability, status
        )):
            raise ValueError(f"Dataset group {group_index} has missing label metadata.")
        if group_id in seen_groups or expected_splits.get(group_id) != split:
            raise ValueError(f"Group {group_id!r} has a duplicate or unstable split.")
        if status not in {"pending_review", "reviewed"}:
            raise ValueError(f"Group {group_id} has invalid review status {status!r}.")
        if answerability not in {"answerable", "unanswerable"}:
            raise ValueError(f"Group {group_id} has invalid answerability.")
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"Group {group_id} must have question variants.")
        if not isinstance(evidence_payload, list) or not isinstance(scope, Mapping):
            raise ValueError(f"Group {group_id} requires evidence and retrieval scope.")
        if answerability == "answerable":
            if not isinstance(group.get("reference_answer"), str) or not evidence_payload:
                raise ValueError(f"Answerable group {group_id} needs answer and evidence.")
            if group.get("expected_refusal") is not None:
                raise ValueError(f"Answerable group {group_id} cannot have a refusal.")
        elif (
            group.get("reference_answer") is not None
            or not isinstance(group.get("expected_refusal"), str)
            or evidence_payload
        ):
            raise ValueError(
                f"Unanswerable group {group_id} needs a refusal and no positive evidence."
            )

        evidence: list[EvidenceUnit] = []
        for evidence_index, item in enumerate(evidence_payload):
            if not isinstance(item, Mapping):
                raise ValueError(f"Evidence in {group_id} must be an object.")
            doc_id = item.get("document_id")
            version = item.get("document_version")
            source_path = item.get("source_path")
            line_start = item.get("line_start")
            line_end = item.get("line_end")
            text = item.get("text")
            if (
                not isinstance(doc_id, str)
                or not isinstance(version, str)
                or not isinstance(source_path, str)
                or not isinstance(line_start, int)
                or isinstance(line_start, bool)
                or not isinstance(line_end, int)
                or isinstance(line_end, bool)
                or line_start < 1
                or line_end < line_start
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise ValueError(f"Evidence {group_id}:{evidence_index} has invalid coordinates.")
            source_doc = documents_by_key.get((doc_id, version))
            if source_doc is None or source_doc.source_name != source_path:
                raise ValueError(
                    f"Evidence {group_id}:{evidence_index} references an unknown source."
                )
            source_lines = source_doc.content.splitlines()
            if line_end > len(source_lines):
                raise ValueError(f"Evidence {group_id}:{evidence_index} exceeds source lines.")
            cited_source = "\n".join(source_lines[line_start - 1:line_end])
            if text not in cited_source:
                raise ValueError(
                    f"Evidence {group_id}:{evidence_index} does not match the cited source lines."
                )
            source_line_offsets = [0]
            for source_line in source_doc.content.splitlines(keepends=True):
                source_line_offsets.append(source_line_offsets[-1] + len(source_line))
            line_char_start = source_line_offsets[line_start - 1]
            line_char_end = source_line_offsets[line_end]
            exact_evidence_start = source_doc.content.find(
                text, line_char_start, line_char_end
            )
            if exact_evidence_start < 0:
                raise ValueError(
                    f"Evidence {group_id}:{evidence_index} has no exact character span."
                )
            evidence.append(
                EvidenceUnit(
                    evidence_id=f"{group_id}-e{evidence_index + 1}",
                    document_id=doc_id,
                    document_version=version,
                    source_path=source_path,
                    line_start=line_start,
                    line_end=line_end,
                    source_char_start=exact_evidence_start,
                    source_char_end=exact_evidence_start + len(text),
                    text=text,
                )
            )

        _validate_scope(group_id, scope, documents_by_key)
        corpus_scope = scope["corpus_wide"]
        allowed_ids = set(corpus_scope["document_ids"])
        allowed_versions = corpus_scope.get("document_versions")
        for item in evidence:
            if item.document_id not in allowed_ids or (
                allowed_versions is not None
                and item.document_version not in allowed_versions
            ):
                raise ValueError(
                    f"Evidence {item.evidence_id} is outside group {group_id}'s "
                    "legitimate corpus-wide scope."
                )
        within_scope = scope.get("within_document")
        if within_scope is not None and any(
            (item.document_id, item.document_version)
            != (within_scope["document_id"], within_scope["version"])
            for item in evidence
        ):
            raise ValueError(
                f"Evidence in {group_id} conflicts with its within-document scope."
            )
        seen_groups.add(group_id)
        for variant in variants:
            if not isinstance(variant, Mapping):
                raise ValueError(f"Question variant in {group_id} must be an object.")
            question_id = variant.get("id")
            question = variant.get("question")
            if (
                not isinstance(question_id, str)
                or not question_id.strip()
                or not isinstance(question, str)
                or not question.strip()
                or question_id in seen_ids
            ):
                raise ValueError(f"Group {group_id} has an invalid or duplicate question ID.")
            seen_ids.add(question_id)
            questions.append(
                EvaluationQuestion(
                    question_id=question_id,
                    group_id=group_id,
                    split=split,
                    category=category,
                    question=question,
                    answerability=answerability,
                    reference_answer=group.get("reference_answer"),
                    expected_refusal=group.get("expected_refusal"),
                    retrieval_scope=scope,
                    evidence=tuple(evidence),
                    review_status=status,
                )
            )
    if seen_groups != set(expected_splits):
        raise ValueError("Split manifest must assign every dataset group exactly once.")
    if not questions:
        raise ValueError("Evaluation dataset has no question cases.")
    return EvaluationDataset(
        questions=tuple(questions),
        split_manifest=manifest,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _validate_scope(
    group_id: str,
    scope: Mapping[str, Any],
    documents: Mapping[tuple[str, str], Document],
) -> None:
    within = scope.get("within_document")
    corpus = scope.get("corpus_wide")
    if not isinstance(corpus, Mapping):
        raise ValueError(f"Group {group_id} requires corpus-wide retrieval scope.")
    permitted_ids = corpus.get("document_ids")
    permitted_versions = corpus.get("document_versions")
    if (
        not isinstance(permitted_ids, list)
        or not permitted_ids
        or any(not isinstance(item, str) for item in permitted_ids)
        or (
            permitted_versions is not None
            and (
                not isinstance(permitted_versions, list)
                or not permitted_versions
                or any(not isinstance(item, str) for item in permitted_versions)
            )
        )
    ):
        raise ValueError(f"Group {group_id} has invalid corpus-wide scope.")
    known_ids = {doc_id for doc_id, _ in documents}
    if not set(permitted_ids).issubset(known_ids):
        raise ValueError(f"Group {group_id} has unknown permitted document IDs.")
    if permitted_versions is not None and not any(
        doc_id in permitted_ids and version in permitted_versions
        for doc_id, version in documents
    ):
        raise ValueError(f"Group {group_id} permitted scope contains no document versions.")
    if within is not None:
        if not isinstance(within, Mapping):
            raise ValueError(f"Group {group_id} has invalid within-document scope.")
        key = (within.get("document_id"), within.get("version"))
        if key not in documents:
            raise ValueError(f"Group {group_id} has unknown within-document scope.")
        if key[0] not in permitted_ids or (
            permitted_versions is not None and key[1] not in permitted_versions
        ):
            raise ValueError(
                f"Group {group_id} within-document scope is outside corpus scope."
            )


def _configuration_factories(
    include_all_chunkers: bool,
) -> list[tuple[str, dict[str, Any], Any]]:
    configs: list[tuple[str, dict[str, Any], Any]] = [
        (
            "token-window",
            {
                "chunk_size": DEFAULT_TARGET_TOKENS,
                "overlap": 12,
                "tokenizer": "unicode-lexical-approximation",
            },
            lambda doc: TokenWindowChunker(
                chunk_size=DEFAULT_TARGET_TOKENS, overlap=12
            ).chunk(doc),
        ),
        (
            "recursive",
            {
                "max_tokens": DEFAULT_TARGET_TOKENS,
                "separators": ["\n\n", "\n", " "],
            },
            lambda doc: RecursiveChunker(max_tokens=DEFAULT_TARGET_TOKENS).chunk(doc),
        ),
        (
            "structure-aware",
            {"max_tokens": DEFAULT_TARGET_TOKENS},
            lambda doc: StructureAwareChunker(
                max_tokens=DEFAULT_TARGET_TOKENS
            ).chunk(doc),
        ),
    ]
    if include_all_chunkers:
        configs.extend(
            [
                (
                    "paragraph",
                    {"max_tokens": DEFAULT_TARGET_TOKENS},
                    lambda doc: ParagraphChunker(max_tokens=DEFAULT_TARGET_TOKENS).chunk(doc),
                ),
                (
                    "table-aware",
                    {"max_tokens": DEFAULT_TARGET_TOKENS},
                    lambda doc: TableAwareChunker(max_tokens=DEFAULT_TARGET_TOKENS).chunk(doc),
                ),
                (
                    "semantic",
                        {
                            "max_tokens": 180,
                            "threshold": 0.35,
                            "similarity_id": "lexical-jaccard-v1",
                        },
                        lambda doc: SemanticChunker(
                            similarity=_lexical_similarity,
                            threshold=0.35,
                            max_tokens=180,
                            similarity_id="lexical-jaccard-v1",
                        ).chunk(doc),
                ),
                (
                    "parent-child",
                    {"parent_max_tokens": 160, "child_size": 80, "child_overlap": 10},
                    lambda doc: parent_child_chunks(
                        doc, parent_max_tokens=160, child_size=80, child_overlap=10
                    ),
                ),
            ]
        )
    return configs


def run_retrieval_evaluation(
    documents: Sequence[Document],
    questions: Sequence[EvaluationQuestion],
    embedding_provider: EmbeddingProvider,
    *,
    mode: str = "corpus-wide",
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    context_budget_tokens: int = DEFAULT_CONTEXT_BUDGET_TOKENS,
    success_k: int = DEFAULT_SUCCESS_K,
    include_all_chunkers: bool = False,
    allow_pending: bool = False,
    split: str = "dev",
    dataset_sha256: str | None = None,
    confirm_held_out: bool = False,
) -> dict[str, Any]:
    """Compare retrieval methods under a single explicit retrieval mode."""
    if mode not in {"within-document", "corpus-wide"}:
        raise ValueError("mode must be 'within-document' or 'corpus-wide'.")
    if split not in {"dev", "held_out"}:
        raise ValueError("split must be 'dev' or 'held_out'.")
    if split == "held_out" and not confirm_held_out:
        raise ValueError(
            "Held-out evaluation requires explicit final-release confirmation."
        )
    if split == "held_out" and allow_pending:
        raise ValueError("Held-out evaluation never allows pending labels.")
    if candidate_limit < 1 or context_budget_tokens < 1 or success_k < 1:
        raise ValueError("Candidate limit, context budget, and success K must be positive.")
    if not questions or not documents:
        raise ValueError("At least one document and one question are required.")
    pending = [question.question_id for question in questions if question.review_status != "reviewed"]
    if pending and not allow_pending:
        raise ValueError(
            "Final benchmark requires reviewed labels; use explicit exploratory mode "
            "to run pending labels."
        )
    if any(question.split != split for question in questions):
        raise ValueError("Evaluation questions must belong to the requested split.")

    documents_by_key = {(doc.document_id, doc.version): doc for doc in documents}
    factories = _configuration_factories(include_all_chunkers)
    cache = _EmbeddingCache(embedding_provider)
    configurations: list[dict[str, Any]] = []
    chunks_by_strategy: dict[str, list[DocumentChunk]] = {}
    parents_by_strategy: dict[str, dict[str, DocumentChunk]] = {}
    for strategy, parameters, factory in factories:
        chunks: list[DocumentChunk] = []
        for document in documents:
            chunks.extend(factory(document))
        chunks_by_strategy[strategy] = chunks
        parents_by_strategy[strategy] = {
            chunk.chunk_id: chunk
            for chunk in chunks
            if chunk.metadata.get("role") == "parent"
        }
        configurations.append(
            {
                "strategy": strategy,
                "parameters": parameters,
                "fingerprint_sha256": chunking_configuration_fingerprint(
                    strategy, parameters
                ),
            }
        )

    corpus_embedding_started = time.perf_counter()
    chunk_vectors_by_strategy = {
        strategy: {
            chunk.chunk_id: cache.embed(chunk.text)
            for chunk in chunks
        }
        for strategy, chunks in chunks_by_strategy.items()
    }
    corpus_embedding_preparation_seconds = (
        time.perf_counter() - corpus_embedding_started
    )

    query_vectors: dict[str, tuple[float, ...]] = {}
    query_embedding_seconds: dict[str, float] = {}
    for question in questions:
        started = time.perf_counter()
        query_vectors[question.question_id] = cache.embed(question.question)
        query_embedding_seconds[question.question_id] = time.perf_counter() - started

    results: list[dict[str, Any]] = []
    all_case_reports: list[dict[str, Any]] = []
    for strategy, _, _ in factories:
        stats: dict[str, dict[str, Any]] = {
            method: {
                "cases": [],
                "recall5": [],
                "recall10": [],
                "mrr10": [],
                "all": [],
                "context_recall5": [],
                "context_recall10": [],
                "context_mrr10": [],
                "context_all": [],
                "child_recall5": [],
                "child_recall10": [],
                "child_mrr10": [],
                "child_all": [],
            }
            for method in INITIAL_METHODS
        }
        chunks = chunks_by_strategy[strategy]
        parents = parents_by_strategy[strategy]
        for question in questions:
            permitted = _permitted_sources(question, mode, documents_by_key)
            if permitted is None:
                continue
            selected = [
                chunk
                for chunk in chunks
                if (chunk.document_id, chunk.document_version) in permitted
                and (strategy != "parent-child" or chunk.metadata.get("role") == "child")
            ]
            selected.sort(key=lambda chunk: chunk.chunk_id)
            truncated = len(selected) > candidate_limit
            candidate_pool = selected[:candidate_limit]
            row_by_id = {chunk.chunk_id: _chunk_row(chunk) for chunk in candidate_pool}
            query_vector = query_vectors[question.question_id]
            query_seconds = query_embedding_seconds[question.question_id]
            started = time.perf_counter()
            dense_ranked = sorted(
                (
                    (
                        cosine_similarity(
                            query_vector,
                            chunk_vectors_by_strategy[strategy][chunk.chunk_id],
                        ),
                        chunk,
                    )
                    for chunk in candidate_pool
                ),
                key=lambda pair: (-pair[0], pair[1].chunk_id),
            )
            dense_seconds = time.perf_counter() - started
            dense_rows = _ranked_chunks_to_rows(dense_ranked)

            started = time.perf_counter()
            bm25_rows = bm25_rank(
                question.question,
                list(row_by_id.values()),
                limit=max(1, candidate_limit),
            )
            bm25_seconds = time.perf_counter() - started
            bm25_ranked = _rows_to_ranked_chunks(bm25_rows)

            started = time.perf_counter()
            rrf_ranked = _rows_to_ranked_chunks(
                reciprocal_rank_fusion(
                    dense_rows,
                    bm25_rows,
                    limit=max(1, candidate_limit),
                )
            )
            rrf_seconds = time.perf_counter() - started

            rerank_pool = candidate_union(
                dense_rows, bm25_rows, candidate_limit=candidate_limit
            )
            started = time.perf_counter()
            reranked = _rows_to_ranked_chunks(
                EmbeddingReranker(_CachedProvider(cache)).rerank(
                    question.question,
                    rerank_pool,
                    limit=max(1, len(rerank_pool)),
                    query_embedding=query_vector,
                )
            )
            rerank_seconds = time.perf_counter() - started

            rankings = {
                "dense": (dense_ranked, dense_seconds, 0.0),
                "bm25": (bm25_ranked, bm25_seconds, 0.0),
                "rrf": (rrf_ranked, dense_seconds + bm25_seconds + rrf_seconds, 0.0),
                "rerank": (
                    reranked,
                    dense_seconds + bm25_seconds,
                    rerank_seconds,
                ),
            }
            for method, (ranking, retrieval_seconds, rerank_seconds_for_method) in rankings.items():
                case_result = _evaluate_case(
                    question=question,
                    ranking=ranking,
                    parents=parents,
                    candidate_pool_size=len(selected),
                    candidate_limit=candidate_limit,
                    truncated=truncated,
                    mode=mode,
                    permitted=permitted,
                    context_budget_tokens=context_budget_tokens,
                    success_k=success_k,
                    query_embedding_seconds=query_seconds if method != "bm25" else 0.0,
                    retrieval_seconds=retrieval_seconds,
                    reranking_seconds=rerank_seconds_for_method,
                )
                stats[method]["cases"].append(case_result)
                if question.answerability == "answerable":
                    metrics = case_result["metrics"]
                    stats[method]["recall5"].append(metrics["evidence_recall_at_5"])
                    stats[method]["recall10"].append(metrics["evidence_recall_at_10"])
                    stats[method]["mrr10"].append(metrics["mrr_at_10"])
                    stats[method]["all"].append(
                        metrics[f"all_required_evidence_success_at_{success_k}"]
                    )
                    context_metrics = case_result["context_metrics"]
                    stats[method]["context_recall5"].append(
                        context_metrics["evidence_recall_at_5"]
                    )
                    stats[method]["context_recall10"].append(
                        context_metrics["evidence_recall_at_10"]
                    )
                    stats[method]["context_mrr10"].append(
                        context_metrics["mrr_at_10"]
                    )
                    stats[method]["context_all"].append(
                        context_metrics[
                            f"all_required_evidence_success_at_{success_k}"
                        ]
                    )
                    child_metrics = case_result["child_ranking_metrics"]
                    if child_metrics is not None:
                        stats[method]["child_recall5"].append(
                            child_metrics["evidence_recall_at_5"]
                        )
                        stats[method]["child_recall10"].append(
                            child_metrics["evidence_recall_at_10"]
                        )
                        stats[method]["child_mrr10"].append(
                            child_metrics["mrr_at_10"]
                        )
                        stats[method]["child_all"].append(
                            child_metrics[
                                f"all_required_evidence_success_at_{success_k}"
                            ]
                        )
                all_case_reports.append(
                    {"strategy": strategy, "method": method, **case_result}
                )

        methods_report: dict[str, Any] = {}
        for method, method_stats in stats.items():
            case_reports = method_stats["cases"]
            answerable_count = len(method_stats["recall5"])
            methods_report[method] = {
                "sample_count": answerable_count,
                "unanswerable_count": sum(
                    case["answerability"] == "unanswerable" for case in case_reports
                ),
                "metrics": {
                    "evidence_recall_at_5": _mean_or_none(method_stats["recall5"]),
                    "evidence_recall_at_10": _mean_or_none(method_stats["recall10"]),
                    "mrr_at_10": _mean_or_none(method_stats["mrr10"]),
                    f"all_required_evidence_success_at_{success_k}": _mean_or_none(
                        method_stats["all"]
                    ),
                },
                "context_metrics_after_budget": {
                    "sample_count": answerable_count,
                    "evidence_recall_at_5": _mean_or_none(
                        method_stats["context_recall5"]
                    ),
                    "evidence_recall_at_10": _mean_or_none(
                        method_stats["context_recall10"]
                    ),
                    "mrr_at_10": _mean_or_none(method_stats["context_mrr10"]),
                    f"all_required_evidence_success_at_{success_k}": _mean_or_none(
                        method_stats["context_all"]
                    ),
                },
                "child_ranking_metrics": {
                    "sample_count": len(method_stats["child_recall5"]),
                    "evidence_recall_at_5": _mean_or_none(
                        method_stats["child_recall5"]
                    ),
                    "evidence_recall_at_10": _mean_or_none(
                        method_stats["child_recall10"]
                    ),
                    "mrr_at_10": _mean_or_none(method_stats["child_mrr10"]),
                    f"all_required_evidence_success_at_{success_k}": _mean_or_none(
                        method_stats["child_all"]
                    ),
                },
                "timing_seconds": _timing_summary(case_reports),
                "queries": case_reports,
            }
        results.append(
            {
                "strategy": strategy,
                "configuration": next(
                    config for config in configurations if config["strategy"] == strategy
                ),
                "chunk_count": len(chunks),
                "methodologies": methods_report,
            }
        )

    actual_case_ids = {
        case["query_id"] for case in all_case_reports
    }
    return {
        "schema_version": 2,
        "git_revision": _git_revision(),
        "git_worktree_dirty": _git_worktree_dirty(),
        "benchmark_mode": mode,
        "split": split,
        "question_count": len(questions),
        "review_status": {
            "reviewed_case_count": sum(q.review_status == "reviewed" for q in questions),
            "pending_review_case_count": len(pending),
            "exploratory_pending_labels": bool(pending and allow_pending),
        },
        "dataset_sha256": dataset_sha256,
        "corpus_sha256": _corpus_fingerprint(documents),
        "embedding": {
            "model": embedding_provider.model,
            "dimensions": embedding_provider.dimensions,
            "configuration_version": embedding_provider.configuration_version,
        },
        "candidate_limit": candidate_limit,
        "context_budget_tokens": context_budget_tokens,
        "success_k": success_k,
        "evidence_matching": {
            "source_coordinates": "one-based inclusive source lines",
            "matching_coordinates": "zero-based half-open source characters derived from the exact gold text",
            "source_version_match_required": True,
            "minimum_source_span_coverage_ratio": EVIDENCE_SPAN_COVERAGE,
            "count_each_evidence_unit_once": True,
        },
        "metric_definitions": {
            "evidence_recall_at_5": "Unique gold evidence units covered by top 5 divided by all gold evidence units.",
            "evidence_recall_at_10": "Unique gold evidence units covered by top 10 divided by all gold evidence units.",
            "mrr_at_10": "Reciprocal rank of the first result matching any gold evidence, truncated at rank 10; it does not prove complete multi-passage support.",
            f"all_required_evidence_success_at_{success_k}": f"1 only when every required evidence unit is covered by at least one of the top {success_k} results; otherwise 0.",
            "unanswerable": "Counted separately; excluded from positive-evidence metrics. No refusal accuracy is measured.",
        },
        "timing_boundaries": {
            "query_embedding": "Wall time for obtaining the query vector; shared by dense/hybrid methods and zero for BM25.",
            "retrieval": "Ranking and fusion wall time using precomputed corpus vectors; includes first-stage retrieval for reranking.",
            "reranking": "Heading-enriched embedding rerank wall time only.",
            "total": "Query embedding + retrieval + reranking; excludes chunking, corpus embedding preparation, database setup, and report writing.",
            "corpus_embedding_preparation": "Wall time to generate/cache chunk vectors before per-query timers start; reported separately.",
            "cold_warm": "Embedding calls and cache hits are reported globally; query vectors and chunk vectors are reused within one run.",
        },
        "embedding_measurements": {
            "actual_embedding_calls": cache.calls,
            "cache_hits": cache.cache_hits,
            "cold_calls": cache.cold_calls,
            "warm_hits": cache.warm_hits,
            "provider_call_seconds": round(cache.call_seconds, 6),
            "warm_cache_lookup_seconds": round(cache.cache_lookup_seconds, 6),
            "corpus_embedding_preparation_seconds": round(
                corpus_embedding_preparation_seconds, 6
            ),
            "provider_token_usage": cache.usage or None,
        },
        "searched_cases": sorted(actual_case_ids),
        "excluded_mode_incompatible_case_count": len(questions) - len(actual_case_ids),
        "strategies": results,
    }


def _permitted_sources(
    question: EvaluationQuestion,
    mode: str,
    documents: Mapping[tuple[str, str], Document],
) -> set[tuple[str, str]] | None:
    if mode == "within-document":
        scope = question.retrieval_scope.get("within_document")
        if scope is None:
            return None
        return {(str(scope["document_id"]), str(scope["version"]))}
    scope = question.retrieval_scope["corpus_wide"]
    ids = set(scope["document_ids"])
    versions = scope.get("document_versions")
    return {
        (doc_id, version)
        for doc_id, version in documents
        if doc_id in ids and (versions is None or version in versions)
    }


def _lexical_similarity(left: str, right: str) -> float:
    left_terms = set(left.casefold().split())
    right_terms = set(right.casefold().split())
    union = left_terms | right_terms
    return len(left_terms & right_terms) / len(union) if union else 1.0


def _evaluate_case(
    *,
    question: EvaluationQuestion,
    ranking: Sequence[tuple[float, DocumentChunk]],
    parents: Mapping[str, DocumentChunk],
    candidate_pool_size: int,
    candidate_limit: int,
    truncated: bool,
    mode: str,
    permitted: set[tuple[str, str]],
    context_budget_tokens: int,
    success_k: int,
    query_embedding_seconds: float,
    retrieval_seconds: float,
    reranking_seconds: float,
) -> dict[str, Any]:
    context_chunks: list[DocumentChunk] = []
    context_ranked: list[tuple[int, DocumentChunk]] = []
    context_tokens = 0
    seen_context_ids: set[str] = set()
    for rank, (_, chunk) in enumerate(ranking, start=1):
        context = parents.get(chunk.parent_chunk_id or "", chunk)
        if context.chunk_id in seen_context_ids:
            continue
        if context_tokens + context.token_count > context_budget_tokens:
            break
        context_chunks.append(context)
        context_ranked.append((rank, context))
        seen_context_ids.add(context.chunk_id)
        context_tokens += context.token_count

    ranking_chunks = [
        (rank, chunk) for rank, (_, chunk) in enumerate(ranking, start=1)
    ]
    evidence_ranks = _match_evidence_ranks(
        question, ranking_chunks, parents, permitted, expand_parents=True
    )
    context_evidence_ranks = _match_evidence_ranks(
        question, context_ranked, {}, permitted, expand_parents=False
    )
    answerable = question.answerability == "answerable"
    metrics = _metrics_for_ranks(evidence_ranks, answerable, success_k)
    context_metrics = _metrics_for_ranks(
        context_evidence_ranks, answerable, success_k
    )
    child_evidence_ranks = None
    child_ranking_metrics = None
    if any(chunk.parent_chunk_id for _, chunk in ranking_chunks):
        child_evidence_ranks = _match_evidence_ranks(
            question, ranking_chunks, {}, permitted, expand_parents=False
        )
        child_ranking_metrics = _metrics_for_ranks(
            child_evidence_ranks, answerable, success_k
        )
    evidence_matches_by_rank: dict[int, list[str]] = {}
    for evidence_id, rank in evidence_ranks.items():
        if rank is not None:
            evidence_matches_by_rank.setdefault(rank, []).append(evidence_id)

    top_chunks = []
    for rank, (score, chunk) in enumerate(ranking[:10], start=1):
        top_chunks.append(
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_version": chunk.document_version,
                "source_name": chunk.source_name,
                "source_line_start": chunk.source_line_start,
                "source_line_end": chunk.source_line_end,
                "source_char_start": chunk.source_char_start,
                "source_char_end": chunk.source_char_end,
                "source_char_spans_used_for_matching": [
                    {"char_start": start, "char_end": end}
                    for start, end in _source_char_spans(chunk)
                ],
                "parent_chunk_id": chunk.parent_chunk_id,
                "expanded_context_chunk_id": (
                    parents[chunk.parent_chunk_id].chunk_id
                    if chunk.parent_chunk_id in parents
                    else None
                ),
                "text": chunk.text,
                "score": round(float(score), 8),
                "matched_evidence_ids": evidence_matches_by_rank.get(rank, []),
            }
        )

    return {
        "query_id": question.question_id,
        "group_id": question.group_id,
        "category": question.category,
        "answerability": question.answerability,
        "reference_answer": question.reference_answer,
        "expected_refusal": question.expected_refusal,
        "review_status": question.review_status,
        "retrieval_mode": mode,
        "searched_document_versions": [
            {"document_id": doc_id, "version": version}
            for doc_id, version in sorted(permitted)
        ],
        "gold_evidence": [
            {
                "evidence_id": evidence.evidence_id,
                "document_id": evidence.document_id,
                "document_version": evidence.document_version,
                "source_path": evidence.source_path,
                "line_start": evidence.line_start,
                "line_end": evidence.line_end,
                "source_char_start": evidence.source_char_start,
                "source_char_end": evidence.source_char_end,
                "text": evidence.text,
            }
            for evidence in question.evidence
        ],
        "candidate_pool": {
            "observed_count": candidate_pool_size,
            "limit": candidate_limit,
            "returned_count": min(candidate_pool_size, candidate_limit),
            "truncated": truncated,
            "complete_coverage": not truncated,
            "order_note": "Candidates are sorted by stable chunk ID before the cap; a truncated BM25 pool is not a corpus-wide lexical search.",
        },
        "evidence_ranks": evidence_ranks,
        "metrics": metrics,
        "context_evidence_ranks": context_evidence_ranks,
        "context_metrics": context_metrics,
        "child_evidence_ranks": child_evidence_ranks,
        "child_ranking_metrics": child_ranking_metrics,
        "context": {
            "budget_tokens": context_budget_tokens,
            "expanded_context_tokens": context_tokens,
            "expanded_context_chunk_count": len(context_chunks),
            "expansion_policy": "Use parent span for evidence matching and count each expanded parent once; stop before exceeding the shared token budget.",
        },
        "timing_seconds": {
            "query_embedding": round(query_embedding_seconds, 6),
            "retrieval": round(retrieval_seconds, 6),
            "reranking": round(reranking_seconds, 6),
            "total_measured_query": round(
                query_embedding_seconds + retrieval_seconds + reranking_seconds, 6
            ),
        },
        "top_chunks": top_chunks,
    }


def _match_evidence_ranks(
    question: EvaluationQuestion,
    ranked_chunks: Sequence[tuple[int, DocumentChunk]],
    parents: Mapping[str, DocumentChunk],
    permitted: set[tuple[str, str]],
    *,
    expand_parents: bool,
) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {
        evidence.evidence_id: None for evidence in question.evidence
    }
    for rank, chunk in ranked_chunks:
        effective = (
            parents.get(chunk.parent_chunk_id or "", chunk)
            if expand_parents
            else chunk
        )
        if (effective.document_id, effective.document_version) not in permitted:
            continue
        for evidence in question.evidence:
            if ranks[evidence.evidence_id] is not None:
                continue
            if (evidence.document_id, evidence.document_version) != (
                effective.document_id,
                effective.document_version,
            ):
                continue
            if (
                _covered_characters(
                    effective, evidence.source_char_start, evidence.source_char_end
                )
                / (evidence.source_char_end - evidence.source_char_start)
                >= EVIDENCE_SPAN_COVERAGE
            ):
                ranks[evidence.evidence_id] = rank
    return ranks


def _metrics_for_ranks(
    evidence_ranks: Mapping[str, int | None],
    answerable: bool,
    success_k: int,
) -> dict[str, float | int] | None:
    if not answerable:
        return None
    if not evidence_ranks:
        raise ValueError("Answerable queries must have positive evidence labels.")

    def recall_at(k: int) -> float:
        return sum(
            rank is not None and rank <= k for rank in evidence_ranks.values()
        ) / len(evidence_ranks)

    first_rank = min(
        (rank for rank in evidence_ranks.values() if rank is not None),
        default=None,
    )
    return {
        "evidence_recall_at_5": recall_at(5),
        "evidence_recall_at_10": recall_at(10),
        "mrr_at_10": (
            1 / first_rank if first_rank is not None and first_rank <= 10 else 0.0
        ),
        f"all_required_evidence_success_at_{success_k}": int(
            all(
                rank is not None and rank <= success_k
                for rank in evidence_ranks.values()
            )
        ),
    }


def _chunk_row(chunk: DocumentChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "heading_path": list(chunk.heading_path),
        "source_char_start": chunk.source_char_start,
        "source_char_end": chunk.source_char_end,
        "source_line_start": chunk.source_line_start,
        "source_line_end": chunk.source_line_end,
        "document_id": chunk.document_id,
        "document_version": chunk.document_version,
        "source_name": chunk.source_name,
        "parent_chunk_id": chunk.parent_chunk_id,
        "_chunk": chunk,
    }


def _source_char_spans(chunk: DocumentChunk) -> list[tuple[int, int]]:
    spans = [(chunk.source_char_start, chunk.source_char_end)]
    header_source = chunk.metadata.get("table_header_source")
    if isinstance(header_source, Mapping):
        start = header_source.get("char_start")
        end = header_source.get("char_end")
        if (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and start >= 0
            and end >= start
        ):
            spans.append((start, end))
    return spans


def _covered_characters(
    chunk: DocumentChunk, evidence_start: int, evidence_end: int
) -> int:
    intersections = sorted(
        (max(start, evidence_start), min(end, evidence_end))
        for start, end in _source_char_spans(chunk)
        if max(start, evidence_start) < min(end, evidence_end)
    )
    covered = 0
    current_start: int | None = None
    current_end: int | None = None
    for start, end in intersections:
        if current_end is None or start > current_end:
            if current_start is not None and current_end is not None:
                covered += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None and current_end is not None:
        covered += current_end - current_start
    return covered


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
    return [(float(row["retrieval_score"]), row["_chunk"]) for row in rows]


def _mean_or_none(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def _timing_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    return {
        "sample_count": len(cases),
        **{
            key: _mean_or_none(
                [float(case["timing_seconds"][key]) for case in cases]
            )
            or 0.0
            for key in (
                "query_embedding",
                "retrieval",
                "reranking",
                "total_measured_query",
            )
        },
    }


def _corpus_fingerprint(documents: Sequence[Document]) -> str:
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: (item.document_id, item.version)):
        digest.update(document.document_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.version.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.source_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            json.dumps(
                document.metadata, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(document.content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _git_worktree_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def write_evaluation_report(report: Mapping[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "retrieval-validation.json"
    markdown_path = destination / "retrieval-validation.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Retrieval validation report",
        "",
        f"- Git revision: `{report.get('git_revision')}`",
        f"- Mode / split: `{report.get('benchmark_mode')}` / `{report.get('split')}`",
        f"- Dataset SHA-256: `{report.get('dataset_sha256')}`",
        f"- Corpus SHA-256: `{report.get('corpus_sha256')}`",
        f"- Review status: `{report.get('review_status')}`",
        f"- Embedding: `{report.get('embedding')}`",
        f"- Candidate limit: `{report.get('candidate_limit')}`; context budget: `{report.get('context_budget_tokens')}` tokens",
        "",
        "Unanswerable cases are counted separately and excluded from evidence metrics. This report does not measure refusal accuracy. MRR@10 measures the first relevant result only and does not establish complete support for multi-passage questions.",
        "",
        "## Aggregate results",
        "",
        "| Chunker | Method | N answerable | Unanswerable | Evidence Recall@5 | Evidence Recall@10 | MRR@10 | All required @K | Budgeted context Recall@5 | Child-rank Recall@5 | Mean total query seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def display_metric(value: Any) -> Any:
        return "N/A" if value is None else value

    for strategy in report.get("strategies", []):
        for method, result in strategy["methodologies"].items():
            metrics = result["metrics"]
            lines.append(
                "| {strategy} | {method} | {n} | {unanswerable} | {r5} | {r10} | {mrr} | {all_required} | {context_r5} | {child_r5} | {timing} |".format(
                    strategy=strategy["strategy"],
                    method=method,
                    n=result["sample_count"],
                    unanswerable=result["unanswerable_count"],
                    r5=display_metric(metrics["evidence_recall_at_5"]),
                    r10=display_metric(metrics["evidence_recall_at_10"]),
                    mrr=display_metric(metrics["mrr_at_10"]),
                    all_required=display_metric(
                        metrics[
                            f"all_required_evidence_success_at_{report['success_k']}"
                        ]
                    ),
                    context_r5=display_metric(
                        result["context_metrics_after_budget"][
                            "evidence_recall_at_5"
                        ]
                    ),
                    child_r5=display_metric(
                        result["child_ranking_metrics"]["evidence_recall_at_5"]
                    ),
                    timing=result["timing_seconds"]["total_measured_query"],
                )
            )
    lines.extend(["", "## Per-query evidence", ""])
    for strategy in report.get("strategies", []):
        for method, result in strategy["methodologies"].items():
            lines.append(f"### {strategy['strategy']} + {method}")
            lines.append("")
            for case in result["queries"]:
                lines.append(
                    f"- `{case['query_id']}` ({case['answerability']}): searched "
                    f"{case['searched_document_versions']}; evidence ranks "
                    f"`{case['evidence_ranks']}`; candidates `{case['candidate_pool']}`."
                )
                for chunk in case["top_chunks"]:
                    lines.append(
                        f"  - Rank {chunk['rank']}: `{chunk['document_id']}@{chunk['document_version']}` "
                        f"lines {chunk['source_line_start']}-{chunk['source_line_end']}, "
                        f"evidence matches `{chunk['matched_evidence_ids']}` — {chunk['text']!r}"
                    )
            lines.append("")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval validation benchmark.")
    parser.add_argument("--run-live", action="store_true", help="Explicitly allow real Gemini embedding calls.")
    parser.add_argument("--mode", choices=("within-document", "corpus-wide"), default="corpus-wide")
    parser.add_argument("--split", choices=("dev", "held_out"), default="dev")
    parser.add_argument("--confirm-held-out", action="store_true")
    parser.add_argument("--exploratory", action="store_true", help="Allow pending human-reviewed labels.")
    parser.add_argument("--include-all-chunkers", action="store_true")
    parser.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
    parser.add_argument("--context-budget-tokens", type=int, default=DEFAULT_CONTEXT_BUDGET_TOKENS)
    parser.add_argument("--success-k", type=int, default=DEFAULT_SUCCESS_K)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "retrieval-validation")
    args = parser.parse_args()
    if not args.run_live:
        parser.error("Live embeddings are opt-in; pass --run-live to authorize provider calls.")
    if args.split == "held_out" and not args.confirm_held_out:
        parser.error("Held-out evaluation requires --confirm-held-out and is reserved for final release.")
    if args.split == "held_out" and args.exploratory:
        parser.error("Exploratory runs cannot evaluate the held-out split.")

    dataset = load_evaluation_dataset()
    questions = tuple(question for question in dataset.questions if question.split == args.split)
    if not args.exploratory and any(
        question.review_status != "reviewed" for question in questions
    ):
        parser.error(
            "Selected labels are pending human review; complete review or use "
            "--exploratory for an explicitly exploratory run."
        )
    report = run_retrieval_evaluation(
        load_policy_corpus(),
        questions,
        GeminiEmbeddingClient(get_settings()),
        mode=args.mode,
        candidate_limit=args.candidate_limit,
        context_budget_tokens=args.context_budget_tokens,
        success_k=args.success_k,
        include_all_chunkers=args.include_all_chunkers,
        allow_pending=args.exploratory,
        split=args.split,
        dataset_sha256=dataset.sha256,
        confirm_held_out=args.confirm_held_out,
    )
    json_path, markdown_path = write_evaluation_report(report, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
