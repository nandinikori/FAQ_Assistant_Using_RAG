from typing import cast

import pytest

from app.retrieval.retrieval import DocumentRetriever, RetrievalMethod
from app.services.rag_service import RAGService


class FakeEmbeddingProvider:
    model = "test-model"
    dimensions = 2
    configuration_version = "test-v1"

    def __init__(self):
        self.calls = []

    def generate_embedding(self, text):
        self.calls.append(text)
        if "domestic" in text.casefold():
            return [1.0, 0.0]
        if "international" in text.casefold():
            return [0.0, 1.0]
        return [0.5, 0.5]


class FakeChunkRepository:
    def __init__(self):
        self.rows = [
            {
                "chunk_id": "domestic",
                "text": "Domestic lodging is USD 220.",
                "heading_path": ["Travel", "Limits"],
                "similarity": 0.7,
            },
            {
                "chunk_id": "international",
                "text": "International lodging is USD 300.",
                "heading_path": ["Travel", "Limits"],
                "similarity": 0.9,
            },
        ]
        self.search_calls = []
        self.list_calls = []

    def search_similar(
        self,
        embedding,
        limit=5,
        document_id=None,
        document_version=None,
        strategy=None,
    ):
        self.search_calls.append(
            (embedding, limit, document_id, document_version, strategy)
        )
        return list(reversed(self.rows))[:limit]

    def list_candidates(
        self, limit=1000, document_id=None, document_version=None, strategy=None
    ):
        self.list_calls.append((limit, document_id, document_version, strategy))
        return self.rows[:limit]


def test_document_retriever_preserves_dense_default_and_optional_strategy_filter():
    repository = FakeChunkRepository()
    embeddings = FakeEmbeddingProvider()
    retriever = DocumentRetriever(repository, embeddings, candidate_limit=10)

    matches = retriever.retrieve(
        "domestic lodging",
        embedding=[1.0, 0.0],
        limit=1,
        document_id="travel",
        document_version="1",
        strategy="table-aware",
    )

    assert matches[0]["chunk_id"] == "international"
    assert "retrieval_method" not in matches[0]
    assert repository.search_calls[0][1:] == (1, "travel", "1", "table-aware")
    assert embeddings.calls == []


def test_document_retriever_supports_bm25_and_rrf():
    repository = FakeChunkRepository()
    embeddings = FakeEmbeddingProvider()
    retriever = DocumentRetriever(repository, embeddings, candidate_limit=10)

    lexical = retriever.retrieve("domestic lodging", method="bm25", limit=1)
    hybrid = retriever.retrieve(
        "domestic lodging",
        method="rrf",
        embedding=[1.0, 0.0],
        limit=2,
    )

    assert lexical[0]["chunk_id"] == "domestic"
    assert lexical[0]["retrieval_method"] == "bm25"
    assert len(hybrid) == 2
    assert all(row["retrieval_method"] == "rrf" for row in hybrid)
    assert embeddings.calls == []


def test_document_retriever_reranks_candidate_union_with_heading_context():
    repository = FakeChunkRepository()
    embeddings = FakeEmbeddingProvider()
    retriever = DocumentRetriever(repository, embeddings, candidate_limit=10)

    matches = retriever.retrieve(
        "domestic lodging",
        method="rerank",
        embedding=[1.0, 0.0],
        limit=1,
    )

    assert matches[0]["chunk_id"] == "domestic"
    assert matches[0]["retrieval_method"] == "rerank"
    assert any("Section: Travel > Limits" in text for text in embeddings.calls)


def test_rag_service_routes_method_and_skips_embedding_for_bm25():
    service = RAGService.__new__(RAGService)
    service.document_repository = FakeChunkRepository()
    service.embedding_client = FakeEmbeddingProvider()
    service.document_retriever = DocumentRetriever(
        service.document_repository, service.embedding_client, candidate_limit=20
    )

    rows = service.retrieve_document_chunks(
        "domestic lodging", method="bm25", document_id="travel"
    )

    assert rows[0]["chunk_id"] == "domestic"
    assert service.embedding_client.calls == []

    with pytest.raises(ValueError, match="Unsupported retrieval method"):
        service.retrieve_document_chunks(
            "domestic lodging", method=cast(RetrievalMethod, "not-a-method")
        )
