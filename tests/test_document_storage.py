import sys
from types import ModuleType, SimpleNamespace

import pytest

from app.chunking.markdown import parse_markdown
from app.chunking.strategies import (
    RecursiveChunker,
    StructureAwareChunker,
    TableAwareChunker,
)
from app.chunking.types import Document
from app.config.settings import Settings
from app.ingestion.document_ingestion import (
    ChunkIngestionError,
    DocumentIngestionService,
)
from app.infrastructure.document_chunk_repository import DocumentChunkRepository
from app.infrastructure.embedding_client import GeminiEmbeddingClient


def test_chunk_identity_includes_chunking_configuration():
    document = Document("policy", "1", "policy.md", "one two three four five")

    first = RecursiveChunker(max_tokens=2).chunk(document)
    repeat = RecursiveChunker(max_tokens=2).chunk(document)
    changed = RecursiveChunker(max_tokens=3).chunk(document)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in repeat]
    assert set(chunk.chunk_id for chunk in first).isdisjoint(
        chunk.chunk_id for chunk in changed
    )
    assert first[0].metadata["configured_max_tokens"] == 2
    assert first[0].token_count == 2


def test_table_aware_chunk_identity_includes_config_for_prose_blocks():
    document = Document("policy", "1", "policy.md", "A prose policy clause.")

    smaller_limit = TableAwareChunker(max_tokens=2).chunk(document)
    larger_limit = TableAwareChunker(max_tokens=4).chunk(document)

    assert smaller_limit[0].chunk_id != larger_limit[0].chunk_id
    assert smaller_limit[0].metadata["configured_max_tokens"] == 2


def test_structure_chunking_parses_markdown_once(monkeypatch):
    document = Document(
        "policy",
        "1",
        "policy.md",
        "# First\n\nAlpha text.\n\n## Second\n\nBeta text.",
    )
    original_parse = parse_markdown
    calls = 0

    def count_parse(source):
        nonlocal calls
        calls += 1
        return original_parse(source)

    monkeypatch.setattr("app.chunking.strategies.parse_markdown", count_parse)

    chunks = StructureAwareChunker(max_tokens=4).chunk(document)

    assert chunks
    assert calls == 1


class FakeChunkRepository:
    def __init__(self):
        self.settings = Settings(gemini_api_key="test-key", embedding_dimensions=3)
        self.stored_ids = set()
        self.insert_calls = []

    def ensure_schema(self):
        pass

    def existing_chunk_ids(self, chunks):
        return self.stored_ids.intersection(chunk.chunk_id for chunk in chunks)

    def insert_chunks(self, chunks, embeddings):
        self.insert_calls.append((list(chunks), list(embeddings)))
        new_ids = {chunk.chunk_id for chunk in chunks} - self.stored_ids
        self.stored_ids.update(new_ids)
        return len(new_ids)


class FakeEmbeddingClient:
    dimensions = 3
    model = "gemini-embedding-001"
    configuration_version = "gemini-embedding-v1"

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def generate_embedding(self, text):
        self.calls.append(text)
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("embedding provider unavailable")
        return [1.0, 0.0, 0.0]


def test_document_ingestion_is_idempotent_and_skips_existing_embeddings():
    repository = FakeChunkRepository()
    embeddings = FakeEmbeddingClient()
    service = DocumentIngestionService(repository, embeddings)
    document = Document(
        "policy",
        "1",
        "policy.md",
        "# First\n\nAlpha applies.\n\n## Second\n\nBeta applies.",
    )
    chunker = StructureAwareChunker(max_tokens=4)

    first = service.ingest(document, chunker)
    second = service.ingest(document, chunker)

    assert first.inserted_count == len(first.chunk_ids)
    assert second.inserted_count == 0
    assert second.already_present_count == len(first.chunk_ids)
    assert first.chunk_ids == second.chunk_ids
    assert len(embeddings.calls) == len(first.chunk_ids)
    assert len(repository.insert_calls) == 2
    assert repository.insert_calls[1] == ([], [])


def test_document_ingestion_reports_chunk_failures_without_inserting():
    repository = FakeChunkRepository()
    embeddings = FakeEmbeddingClient(fail_on="Beta")
    service = DocumentIngestionService(repository, embeddings)
    document = Document(
        "policy",
        "1",
        "policy.md",
        "# First\n\nAlpha applies.\n\n## Second\n\nBeta applies.",
    )

    with pytest.raises(ChunkIngestionError, match="failed chunks") as error:
        service.ingest(document, StructureAwareChunker(max_tokens=4))

    assert len(error.value.failures) == 1
    assert "embedding provider unavailable" in error.value.failures[0][1]
    assert not repository.insert_calls


class UnusedDatabaseClient:
    def connect(self):
        raise AssertionError("dimension errors must be rejected before connecting")


def test_repository_rejects_wrong_embedding_dimensions_before_database_access():
    settings = Settings(
        gemini_api_key="test-key",
        embedding_dimensions=3,
    )
    repository = DocumentChunkRepository(UnusedDatabaseClient(), settings)
    chunk = StructureAwareChunker().chunk(
        Document("policy", "1", "policy.md", "A policy clause.")
    )[0]

    with pytest.raises(ValueError, match="expected 3"):
        repository.insert_chunks([chunk], [[1.0, 0.0]])


def test_gemini_embedding_client_sends_configured_model_and_dimensions(monkeypatch):
    captured = {}

    class FakeModels:
        def embed_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])]
            )

    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "test-key"
            self.models = FakeModels()

    genai = ModuleType("google.genai")
    genai.Client = FakeClient
    genai.types = SimpleNamespace(
        EmbedContentConfig=lambda **kwargs: SimpleNamespace(**kwargs)
    )
    google = ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    client = GeminiEmbeddingClient(
        Settings(
            gemini_api_key="test-key",
            gemini_embedding_model="test-embedding-model",
            embedding_dimensions=3,
        )
    )

    assert client.generate_embedding("test text") == [0.1, 0.2, 0.3]
    assert captured["model"] == "test-embedding-model"
    assert captured["contents"] == "test text"
    assert captured["config"].output_dimensionality == 3
