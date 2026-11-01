from typing import cast

import pytest

from app.retrieval.retrieval import (
    DocumentRetriever,
    RetrievalMethod,
    candidate_union,
)
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
        chunking_configuration=None,
        document_ids=None,
        document_versions=None,
    ):
        self.search_calls.append(
            (
                embedding,
                limit,
                document_id,
                document_version,
                strategy,
                chunking_configuration,
                document_ids,
                document_versions,
            )
        )
        return list(reversed(self.rows))[:limit]

    def list_candidates(
        self,
        limit=1000,
        document_id=None,
        document_version=None,
        strategy=None,
        chunking_configuration=None,
        document_ids=None,
        document_versions=None,
    ):
        self.list_calls.append(
            (
                limit,
                document_id,
                document_version,
                strategy,
                chunking_configuration,
                document_ids,
                document_versions,
            )
        )
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
    assert repository.search_calls[0][1:5] == (1, "travel", "1", "table-aware")
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


@pytest.mark.parametrize("method", ["dense", "bm25", "rrf", "rerank"])
def test_complete_configuration_and_legitimate_document_scope_propagate(method):
    repository = FakeChunkRepository()
    embeddings = FakeEmbeddingProvider()
    retriever = DocumentRetriever(repository, embeddings, candidate_limit=10)
    configuration = {
        "chunk_size": 80,
        "overlap": 12,
        "tokenizer": "unicode-lexical-approximation",
    }
    scope_ids = ["travel", "remote"]
    scope_versions = ["1.0", "2.0"]

    retriever.retrieve(
        "domestic lodging",
        method=method,
        embedding=None if method == "bm25" else [1.0, 0.0],
        chunking_configuration=configuration,
        document_ids=scope_ids,
        document_versions=scope_versions,
        strategy="token-window",
    )

    assert repository.list_calls or repository.search_calls
    assert all(call[3] == "token-window" for call in repository.list_calls)
    assert all(call[4] == "token-window" for call in repository.search_calls)
    assert all(call[4] == configuration for call in repository.list_calls)
    assert all(call[5] == configuration for call in repository.search_calls)
    assert all(call[5] == scope_ids for call in repository.list_calls)
    assert all(call[6] == scope_ids for call in repository.search_calls)
    assert all(call[6] == scope_versions for call in repository.list_calls)
    assert all(call[7] == scope_versions for call in repository.search_calls)


def test_bm25_candidate_limit_is_observable_and_not_claimed_complete():
    repository = FakeChunkRepository()
    retriever = DocumentRetriever(repository, FakeEmbeddingProvider(), candidate_limit=1)

    rows = retriever.retrieve("domestic lodging", method="bm25", limit=1)

    assert repository.list_calls[0][0] == 2
    assert rows[0]["candidate_pool_limit"] == 1
    assert rows[0]["candidate_pool_truncated"] is True
    assert rows[0]["candidate_pool_observed_count"] == 2


def test_candidate_union_applies_limit_across_both_first_stage_lists():
    dense = [{"chunk_id": "d1"}, {"chunk_id": "d2"}]
    lexical = [{"chunk_id": "l1"}, {"chunk_id": "d1"}]

    union = candidate_union(dense, lexical, candidate_limit=2)

    assert [row["chunk_id"] for row in union] == ["d1", "l1"]


def test_embedding_rerank_includes_unique_bm25_candidates_when_dense_fills_pool():
    class LexicalOnlyRepository(FakeChunkRepository):
        def search_similar(self, embedding, limit=5, **filters):
            return self.rows[:2]

        def list_candidates(self, limit=1000, **filters):
            return [
                {
                    "chunk_id": "lexical-only",
                    "text": "exact needle identifier",
                    "heading_path": [],
                }
            ]

    class RecordingEmbeddings(FakeEmbeddingProvider):
        def generate_embedding(self, text):
            self.calls.append(text)
            if "exact needle" in text:
                return [0.0, 1.0]
            return [1.0, 0.0]

    embeddings = RecordingEmbeddings()
    retriever = DocumentRetriever(
        LexicalOnlyRepository(), embeddings, candidate_limit=2
    )

    rows = retriever.retrieve(
        "exact needle identifier",
        method="rerank",
        embedding=[1.0, 0.0],
        limit=2,
    )

    assert any("exact needle identifier" in text for text in embeddings.calls)
    assert "lexical-only" in {row["chunk_id"] for row in rows}


def test_embedding_rerank_reports_union_truncation():
    retriever = DocumentRetriever(
        FakeChunkRepository(), FakeEmbeddingProvider(), candidate_limit=1
    )

    rows = retriever.retrieve(
        "domestic lodging",
        method="rerank",
        embedding=[1.0, 0.0],
        limit=1,
    )

    assert rows[0]["candidate_pool_limit"] == 1
    assert rows[0]["candidate_pool_truncated"] is True
    assert rows[0]["candidate_pool_observed_count"] == 2


def test_embedding_rerank_retains_first_stage_probe_count_when_only_dense_truncates():
    class OneLexicalCandidateRepository(FakeChunkRepository):
        def list_candidates(self, limit=1000, **filters):
            return [self.rows[1]]

    retriever = DocumentRetriever(
        OneLexicalCandidateRepository(), FakeEmbeddingProvider(), candidate_limit=1
    )

    rows = retriever.retrieve(
        "international lodging",
        method="rerank",
        embedding=[0.0, 1.0],
        limit=1,
    )

    assert rows[0]["candidate_pool_truncated"] is True
    assert rows[0]["candidate_pool_observed_count"] == 2


def test_corpus_scope_is_not_replaced_by_gold_document_filter():
    repository = FakeChunkRepository()
    retriever = DocumentRetriever(repository, FakeEmbeddingProvider())
    allowed_documents = ["travel", "remote"]

    retriever.retrieve(
        "domestic lodging",
        method="rrf",
        embedding=[1.0, 0.0],
        document_ids=allowed_documents,
    )

    assert repository.search_calls[0][2] is None
    assert repository.search_calls[0][6] == allowed_documents
    assert repository.list_calls[0][1] is None
    assert repository.list_calls[0][5] == allowed_documents


def test_rag_service_propagates_complete_configuration_and_scope():
    service = RAGService.__new__(RAGService)
    service.document_repository = FakeChunkRepository()
    service.embedding_client = FakeEmbeddingProvider()
    service.document_retriever = DocumentRetriever(
        service.document_repository, service.embedding_client, candidate_limit=20
    )

    service.retrieve_document_chunks(
        "domestic lodging",
        method="bm25",
        strategy="structure-aware",
        chunking_configuration={"max_tokens": 80},
        document_ids=["travel-expense", "remote-work"],
        document_versions=["1.0", "2.0"],
    )

    call = service.document_repository.list_calls[0]
    assert call[4] == {"max_tokens": 80}
    assert call[5] == ["travel-expense", "remote-work"]
    assert call[6] == ["1.0", "2.0"]
