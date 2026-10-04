import json

import pytest

from app.config.settings import Settings
from app.domain.faq import FAQ
from app.infrastructure.faq_repository import FAQRepository
from app.services.AI_assistant_service import AIAssistantService, NO_MATCH_MESSAGE
from app.services.faq_ingestion_service import FAQIngestionService
from app.services.rag_service import RAGService
from main import handle_ai_answer, handle_retrieval, show_menu


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

    context = service.get_context("How can I store records efficiently?")

    assert "What is a database?" in context
    assert "A database stores structured information." in context


def test_ask_ai_generates_answer_from_retrieved_context():
    class FakeRAGService:
        def get_context(self, question):
            assert question == "What is PostgreSQL?"
            return "Q: What is PostgreSQL?\nA: PostgreSQL is a relational database."

    class FakeModels:
        def generate_content(self, model, contents):
            assert model == "test-model"
            assert "PostgreSQL is a relational database" in contents
            return type("Response", (), {"text": "**PostgreSQL** is a relational database."})()

    class FakeClient:
        models = FakeModels()

    settings = Settings(gemini_api_key="test-key", gemini_text_model="test-model")
    assistant = AIAssistantService(settings, FakeRAGService(), FakeClient())

    assert assistant.ask_ai("What is PostgreSQL?") == "**PostgreSQL** is a relational database."


def test_ask_ai_does_not_generate_when_no_context_is_found():
    class FakeRAGService:
        def get_context(self, question):
            return ""

    class FakeClient:
        models = None

    settings = Settings(gemini_api_key="test-key")
    assistant = AIAssistantService(settings, FakeRAGService(), FakeClient())

    assert assistant.ask_ai("What is PostgreSQL?") == NO_MATCH_MESSAGE


def test_retrieval_menu_handler_prints_matches_without_ai(monkeypatch, capsys):
    class FakeRAGService:
        def __init__(self, settings):
            pass

        def retrieve(self, question):
            assert question == "What is PostgreSQL?"
            return [{
                "question": "What is PostgreSQL?",
                "answer": "PostgreSQL is a relational database.",
                "category": "database",
                "similarity": 0.91,
            }]

    monkeypatch.setattr("main.RAGService", FakeRAGService)
    monkeypatch.setattr("builtins.input", lambda prompt: "What is PostgreSQL?")

    handle_retrieval(object())

    output = capsys.readouterr().out
    assert "Retrieved FAQ matches:" in output
    assert "PostgreSQL is a relational database." in output
    assert "Similarity: 0.910" in output
    assert "AI:" not in output


def test_ai_menu_handler_prints_generated_answer(monkeypatch, capsys):
    class FakeAssistant:
        def __init__(self, settings):
            pass

        def ask_ai(self, question):
            assert question == "What is PostgreSQL?"
            return "**PostgreSQL** is a relational database."

    monkeypatch.setattr("main.AIAssistantService", FakeAssistant)
    monkeypatch.setattr("builtins.input", lambda prompt: "What is PostgreSQL?")

    handle_ai_answer(object())

    output = capsys.readouterr().out
    assert "AI:" in output
    assert "**PostgreSQL** is a relational database." in output


def test_menu_shows_four_workflow_choices(capsys):
    show_menu()

    output = capsys.readouterr().out
    assert "2. RAG retrieval" in output
    assert "3. Ask AI" in output
    assert "4. Exit" in output
