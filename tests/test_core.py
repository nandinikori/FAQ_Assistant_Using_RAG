import json

import pytest

from app.config.settings import Settings
from app.domain.faq import FAQ
from app.infrastructure.faq_repository import FAQRepository
from app.services.faq_ingestion_service import FAQIngestionService
from app.services.rag_service import RAGService


def test_settings_validate_requires_api_key():
    settings = Settings(gemini_api_key="")

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        settings.validate()


def test_faq_to_insert_tuple_uses_expected_fields():
    faq = FAQ(
        question="What is retrieval augmented generation?",
        answer="It combines retrieval and generation to ground model responses.",
        category="concept",
    )

    assert faq.to_insert_tuple() == (
        "What is retrieval augmented generation?",
        "It combines retrieval and generation to ground model responses.",
        "concept",
    )


def test_load_json_supports_list_payload(tmp_path):
    payload_path = tmp_path / "faqs.json"
    payload_path.write_text(
        json.dumps([
            {"question": "What is pgvector?", "answer": "A PostgreSQL extension for vector similarity search.", "category": "database"}
        ]),
        encoding="utf-8",
    )

    service = FAQIngestionService(database_client=object(), embedding_client=object())
    faqs = service.load_json(payload_path)

    assert len(faqs) == 1
    assert faqs[0].question == "What is pgvector?"
    assert faqs[0].answer == "A PostgreSQL extension for vector similarity search."


def test_ingestion_service_embeds_and_stores_faqs():
    class FakeEmbeddingClient:
        def generate_embedding(self, text):
            return [0.1, 0.2, 0.3]

    class FakeRepository:
        def __init__(self):
            self.calls = []

        def ensure_schema(self):
            return None

        def upsert(self, faq, embedding):
            self.calls.append((faq, embedding))

    service = FAQIngestionService(database_client=object(), embedding_client=FakeEmbeddingClient())
    service.repository = FakeRepository()

    faq = FAQ(question="How does RAG work?", answer="It retrieves similar context before answering.", category="concept")
    inserted = service.ingest([faq])

    assert inserted == 1
    assert len(service.repository.calls) == 1
    assert service.repository.calls[0][0].question == "How does RAG work?"
    assert service.repository.calls[0][1] == [0.1, 0.2, 0.3]


def test_repository_upsert_formats_vector_literal():
    class FakeCursor:
        def __init__(self):
            self.query = None
            self.params = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params=()):
            self.query = query
            self.params = params

        def fetchone(self):
            return {"id": 12}

    class FakeConnection:
        def __init__(self):
            self.fake_cursor = FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return self.fake_cursor

    class FakeDatabaseClient:
        def __init__(self):
            self.connection = FakeConnection()

        def connect(self):
            return self.connection

    database_client = FakeDatabaseClient()
    repo = FAQRepository(database_client)
    faq = FAQ(question="What is a vector database?", answer="It stores numerical embeddings for similarity search.", category="database")

    inserted_id = repo.upsert(faq, [1.0, 2.0, 3.0])

    assert inserted_id == 12
    assert database_client.connection.fake_cursor.params == (
        faq.question,
        faq.answer,
        faq.category,
        "[1.0,2.0,3.0]",
    )


def test_rag_service_returns_relevant_faq_context():
    class FakeEmbeddingClient:
        def generate_embedding(self, text):
            return [0.5, 0.6, 0.7]

    class FakeRepository:
        def search_similar(self, embedding, limit=5):
            return [
                {"question": "What is a database?", "answer": "A database stores structured information."},
                {"question": "What is a vector store?", "answer": "A vector store indexes embeddings for search."},
            ]

    service = RAGService.__new__(RAGService)
    service.settings = object()
    service.database_client = object()
    service.embedding_client = FakeEmbeddingClient()
    service.repository = FakeRepository()

    answer = service.answer_question("How can I store records efficiently?")

    assert "What is a database?" in answer
    assert "A database stores structured information." in answer
