"""Retrieval algorithms and vector-search adapters for FAQ and document RAG."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from app.chunking.configuration import validate_chunking_configuration
from app.infrastructure.database import DatabaseClient

RetrievalMethod = Literal["dense", "bm25", "rrf", "rerank"]
_TOKEN = re.compile(r"\w+", re.UNICODE)
DEFAULT_RRF_CONSTANT = 60


class EmbeddingProvider(Protocol):
    def generate_embedding(self, text: str) -> Sequence[float]: ...


class DocumentChunkSearchRepository(Protocol):
    def search_similar(
        self,
        embedding: Sequence[float],
        limit: int = 5,
        document_id: str | None = None,
        document_version: str | None = None,
        strategy: str | None = None,
        chunking_configuration: Mapping[str, Any] | None = None,
        document_ids: Sequence[str] | None = None,
        document_versions: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    def list_candidates(
        self,
        limit: int = 1000,
        document_id: str | None = None,
        document_version: str | None = None,
        strategy: str | None = None,
        chunking_configuration: Mapping[str, Any] | None = None,
        document_ids: Sequence[str] | None = None,
        document_versions: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]: ...


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for equal-length, finite, non-zero vectors."""
    if not left or len(left) != len(right):
        raise ValueError("Cosine similarity requires non-empty vectors of equal length.")
    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    if not all(math.isfinite(value) for value in (*left_values, *right_values)):
        raise ValueError("Cosine similarity requires finite vector values.")
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return sum(a * b for a, b in zip(left_values, right_values, strict=True)) / (
        left_norm * right_norm
    )


def bm25_rank(
    query: str,
    documents: Sequence[Mapping[str, Any]],
    limit: int,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    """Rank candidate chunks with Okapi BM25 and return scored row copies."""
    if limit < 1:
        raise ValueError("limit must be positive.")
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("BM25 requires k1 > 0 and b between 0 and 1.")
    if not documents:
        return []

    tokenized = [
        [match.group(0).casefold() for match in _TOKEN.finditer(str(row["text"]))]
        for row in documents
    ]
    query_terms = set(match.group(0).casefold() for match in _TOKEN.finditer(query))
    if not query_terms:
        return []

    average_length = sum(map(len, tokenized)) / len(tokenized) or 1.0
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    ranked: list[tuple[float, dict[str, Any]]] = []
    for row, tokens in zip(documents, tokenized, strict=True):
        term_counts = Counter(tokens)
        length_norm = k1 * (1 - b + b * len(tokens) / average_length)
        score = 0.0
        for term in query_terms.intersection(term_counts):
            frequency = term_counts[term]
            doc_freq = document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (len(documents) - doc_freq + 0.5) / (doc_freq + 0.5)
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1) / (frequency + length_norm)
            )
        result = dict(row)
        result["retrieval_score"] = score
        result["retrieval_method"] = "bm25"
        ranked.append((score, result))

    ranked.sort(
        key=lambda item: (-item[0], _row_identity(item[1]))
    )
    return [row for _, row in ranked[:limit]]


def reciprocal_rank_fusion(
    *rankings: Sequence[Mapping[str, Any]],
    limit: int,
    rrf_constant: int = DEFAULT_RRF_CONSTANT,
) -> list[dict[str, Any]]:
    """Fuse ranked result lists with RRF, without comparing their raw scores."""
    if limit < 1:
        raise ValueError("limit must be positive.")
    if rrf_constant < 1:
        raise ValueError("RRF constant must be positive.")
    fused_scores: dict[str, float] = {}
    rows_by_id: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            identity = _row_identity(row)
            rows_by_id[identity] = dict(row)
            fused_scores[identity] = fused_scores.get(identity, 0.0) + (
                1 / (rrf_constant + rank)
            )
    ranked_ids = sorted(
        fused_scores,
        key=lambda identity: (-fused_scores[identity], identity),
    )
    result = []
    for identity in ranked_ids[:limit]:
        row = rows_by_id[identity]
        row["retrieval_score"] = fused_scores[identity]
        row["retrieval_method"] = "rrf"
        result.append(row)
    return result


def candidate_union(
    *rankings: Sequence[Mapping[str, Any]],
    candidate_limit: int,
) -> list[dict[str, Any]]:
    """Interleave ranked lists into a stable, deduplicated candidate pool."""
    if candidate_limit < 1:
        raise ValueError("candidate_limit must be positive.")
    candidates: dict[str, dict[str, Any]] = {}
    for rank in range(max((len(ranking) for ranking in rankings), default=0)):
        for ranking in rankings:
            if rank >= len(ranking):
                continue
            row = ranking[rank]
            identity = _row_identity(row)
            if identity in candidates:
                continue
            if len(candidates) == candidate_limit:
                return list(candidates.values())
            candidates[identity] = dict(row)
    return list(candidates.values())


class EmbeddingReranker:
    """Re-score a bounded first-stage candidate list using query/passage cosine."""

    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        limit: int,
        *,
        query_embedding: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive.")
        if not candidates:
            return []
        query_vector = (
            query_embedding
            if query_embedding is not None
            else self.embedding_provider.generate_embedding(query)
        )
        ranked = []
        for row in candidates:
            heading_path = row.get("heading_path") or []
            if isinstance(heading_path, str):
                heading = heading_path
            else:
                heading = " > ".join(str(part) for part in heading_path)
            passage = (
                f"Section: {heading}\nPassage: {row['text']}"
                if heading
                else str(row["text"])
            )
            score = cosine_similarity(
                query_vector, self.embedding_provider.generate_embedding(passage)
            )
            result = dict(row)
            result["retrieval_score"] = score
            result["retrieval_method"] = "rerank"
            ranked.append((score, _row_identity(result), result))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [row for _, _, row in ranked[:limit]]


class DocumentRetriever:
    """Orchestrate dense, BM25, RRF, and embedding-reranked chunk retrieval."""

    def __init__(
        self,
        repository: DocumentChunkSearchRepository,
        embedding_provider: EmbeddingProvider,
        *,
        rrf_constant: int = DEFAULT_RRF_CONSTANT,
        candidate_limit: int = 1000,
    ):
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive.")
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.rrf_constant = rrf_constant
        self.candidate_limit = candidate_limit
        self.reranker = EmbeddingReranker(embedding_provider)

    def retrieve(
        self,
        query: str,
        *,
        method: RetrievalMethod = "dense",
        limit: int = 5,
        embedding: Sequence[float] | None = None,
        document_id: str | None = None,
        document_version: str | None = None,
        strategy: str | None = None,
        candidate_limit: int | None = None,
        chunking_configuration: Mapping[str, Any] | None = None,
        document_ids: Sequence[str] | None = None,
        document_versions: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive.")
        if method not in ("dense", "bm25", "rrf", "rerank"):
            raise ValueError(f"Unsupported retrieval method: {method}")
        candidate_pool_limit = (
            self.candidate_limit if candidate_limit is None else candidate_limit
        )
        if candidate_pool_limit < 1:
            raise ValueError("candidate_limit must be positive.")
        if chunking_configuration is not None and strategy is None:
            raise ValueError(
                "A strategy is required when filtering by chunking configuration."
            )
        if chunking_configuration is not None and strategy is not None:
            validate_chunking_configuration(strategy, chunking_configuration)
        if method == "bm25":
            candidates = self.repository.list_candidates(
                limit=candidate_pool_limit + 1,
                document_id=document_id,
                document_version=document_version,
                strategy=strategy,
                chunking_configuration=chunking_configuration,
                document_ids=document_ids,
                document_versions=document_versions,
            )
            truncated = len(candidates) > candidate_pool_limit
            ranked = bm25_rank(query, candidates[:candidate_pool_limit], limit)
            return _with_candidate_coverage(
                ranked, len(candidates), candidate_pool_limit, truncated
            )

        if embedding is None:
            raise ValueError(f"An embedding is required for {method} retrieval.")
        dense_candidates = self.repository.search_similar(
            embedding,
            limit=max(limit, candidate_pool_limit) + 1 if method != "dense" else limit,
            document_id=document_id,
            document_version=document_version,
            strategy=strategy,
            chunking_configuration=chunking_configuration,
            document_ids=document_ids,
            document_versions=document_versions,
        )
        if method == "dense":
            return dense_candidates[:limit]
        if method == "rrf":
            lexical_candidates = self.repository.list_candidates(
                limit=candidate_pool_limit + 1,
                document_id=document_id,
                document_version=document_version,
                strategy=strategy,
                chunking_configuration=chunking_configuration,
                document_ids=document_ids,
                document_versions=document_versions,
            )
            truncated = (
                len(dense_candidates) > candidate_pool_limit
                or len(lexical_candidates) > candidate_pool_limit
            )
            observed_count = max(len(dense_candidates), len(lexical_candidates))
            dense_candidates = dense_candidates[:candidate_pool_limit]
            lexical = bm25_rank(
                query, lexical_candidates[:candidate_pool_limit], candidate_pool_limit
            )
            ranked = reciprocal_rank_fusion(
                dense_candidates,
                lexical,
                limit=limit,
                rrf_constant=self.rrf_constant,
            )
            return _with_candidate_coverage(
                ranked,
                observed_count,
                candidate_pool_limit,
                truncated,
            )

        lexical_candidates = self.repository.list_candidates(
            limit=candidate_pool_limit + 1,
            document_id=document_id,
            document_version=document_version,
            strategy=strategy,
            chunking_configuration=chunking_configuration,
            document_ids=document_ids,
            document_versions=document_versions,
        )
        truncated = (
            len(dense_candidates) > candidate_pool_limit
            or len(lexical_candidates) > candidate_pool_limit
        )
        observed_count = max(len(dense_candidates), len(lexical_candidates))
        dense_candidates = dense_candidates[:candidate_pool_limit]
        lexical = bm25_rank(
            query, lexical_candidates[:candidate_pool_limit], candidate_pool_limit
        )
        rerank_candidates = candidate_union(
            dense_candidates,
            lexical,
            candidate_limit=candidate_pool_limit + 1,
        )
        union_truncated = len(rerank_candidates) > candidate_pool_limit
        rerank_candidates = rerank_candidates[:candidate_pool_limit]
        ranked = self.reranker.rerank(
            query,
            rerank_candidates,
            limit,
            query_embedding=embedding,
        )
        return _with_candidate_coverage(
            ranked,
            max(
                observed_count,
                candidate_pool_limit + 1 if union_truncated else 0,
            ),
            candidate_pool_limit,
            truncated or union_truncated,
        )


def _row_identity(row: Mapping[str, Any]) -> str:
    identity = row.get("chunk_id", row.get("id"))
    if identity is None:
        raise ValueError("Retrieval rows require a chunk_id or id.")
    return str(identity)


def _with_candidate_coverage(
    rows: list[dict[str, Any]], observed_count: int, limit: int, truncated: bool
) -> list[dict[str, Any]]:
    coverage = {
        "candidate_pool_observed_count": min(observed_count, limit + 1),
        "candidate_pool_limit": limit,
        "candidate_pool_truncated": truncated,
    }
    return [{**row, **coverage} for row in rows]


class FAQRetrieval:
    """Searches the FAQ table for the most similar vector matches."""

    def __init__(self, database_client: DatabaseClient):
        self.database_client = database_client

    def search_similar(self, embedding: Sequence[float], limit: int = 5):
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
