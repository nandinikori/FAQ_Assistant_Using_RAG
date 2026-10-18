import os
from uuid import uuid4

import pytest

from app.chunking.strategies import StructureAwareChunker, parent_child_chunks
from app.chunking.types import Document
from app.config.settings import Settings
from app.domain.faq import FAQ
from app.ingestion.document_ingestion import DocumentIngestionService
from app.infrastructure.database import DatabaseClient
from app.infrastructure.document_chunk_repository import DocumentChunkRepository
from app.infrastructure.faq_repository import FAQRepository


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION_TESTS") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION_TESTS=1 to use configured PostgreSQL.",
)


class FixedEmbeddingClient:
    def __init__(self, model, dimensions, configuration_version):
        self.model = model
        self.dimensions = dimensions
        self.configuration_version = configuration_version

    def generate_embedding(self, _text):
        assert isinstance(_text, str)
        return [1.0] + [0.0] * (self.dimensions - 1)


class ParentChildChunker:
    def chunk(self, document):
        return parent_child_chunks(
            document, parent_max_tokens=20, child_size=5, child_overlap=1
        )


def test_postgres_document_ingestion_retrieval_and_provenance():
    settings = Settings(
        gemini_api_key="integration-test-placeholder",
        db_host=os.getenv("DB_HOST", "127.0.0.1"),
        db_port=int(os.getenv("DB_PORT", "5432")),
        db_name=os.getenv("DB_NAME", "ai_workshop"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", "postgres"),
        embedding_dimensions=3072,
    )
    database = DatabaseClient(settings)
    repository = DocumentChunkRepository(database, settings)
    faq_repository = FAQRepository(database)
    embeddings = FixedEmbeddingClient(
        settings.gemini_embedding_model,
        settings.embedding_dimensions,
        settings.embedding_configuration_version,
    )
    ingestion = DocumentIngestionService(repository, embeddings)
    document_id = f"postgres-integration-{uuid4()}"
    chunker = StructureAwareChunker(max_tokens=100)
    v1 = Document(
        document_id,
        "1.0",
        "remote_work_v1.md",
        "# Remote work\n\nRemote employees may work from home two days each week.",
    )
    v2 = Document(
        document_id,
        "2.0",
        "remote_work_v2.md",
        "# Remote work\n\nRemote employees may work from home three days each week.",
    )

    def stored_chunk_ids(version):
        with database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT chunk_id
                    FROM document_chunks
                    WHERE document_id = %s
                      AND document_version = %s
                      AND embedding_model = %s
                      AND embedding_dimensions = %s
                      AND embedding_configuration_version = %s
                    ORDER BY chunk_id;
                    """,
                    (
                        document_id,
                        version,
                        settings.gemini_embedding_model,
                        settings.embedding_dimensions,
                        settings.embedding_configuration_version,
                    ),
                )
                return [row["chunk_id"] for row in cursor.fetchall()]

    try:
        first_ingestion = ingestion.ingest(v1, chunker)
        ids_after_first_ingestion = stored_chunk_ids("1.0")
        duplicate_ingestion = ingestion.ingest(v1, chunker)
        ids_after_duplicate_ingestion = stored_chunk_ids("1.0")
        v2_ingestion = ingestion.ingest(v2, chunker)

        assert first_ingestion.inserted_count == 1
        assert duplicate_ingestion.inserted_count == 0
        assert first_ingestion.chunk_ids == duplicate_ingestion.chunk_ids
        assert ids_after_first_ingestion == ids_after_duplicate_ingestion
        assert tuple(ids_after_duplicate_ingestion) == first_ingestion.chunk_ids
        assert v2_ingestion.inserted_count == 1
        assert len(stored_chunk_ids("1.0")) == 1
        assert len(stored_chunk_ids("2.0")) == 1

        changed_chunker_result = ingestion.ingest(
            v1, StructureAwareChunker(max_tokens=99)
        )
        assert changed_chunker_result.inserted_count == 1
        assert set(changed_chunker_result.chunk_ids).isdisjoint(
            first_ingestion.chunk_ids
        )
        assert len(stored_chunk_ids("1.0")) == 2
        ingestion.ingest(v1, ParentChildChunker())

        query_embedding = embeddings.generate_embedding(
            "How many days may remote employees work at home?"
        )
        parent_child_results = repository.search_similar(
            query_embedding,
            limit=100,
            document_id=document_id,
            document_version="1.0",
        )
        parent_ids = {
            row["chunk_id"]
            for row in parent_child_results
            if row["metadata"].get("role") == "parent"
        }
        child_rows = [
            row for row in parent_child_results
            if row["metadata"].get("role") == "child"
        ]
        assert child_rows
        assert all(row["parent_chunk_id"] in parent_ids for row in child_rows)

        results = repository.search_similar(
            query_embedding,
            document_id=document_id,
            document_version="2.0",
        )
        assert len(results) == 1
        assert results[0]["document_version"] == "2.0"
        assert "three days" in results[0]["text"]
        assert (
            v2.content[
                results[0]["source_char_start"]:results[0]["source_char_end"]
            ]
            == results[0]["text"]
        )
        assert results[0]["source_name"] == "remote_work_v2.md"
        assert results[0]["source_line_start"] == 1

        with pytest.raises(
            ValueError, match=f"expected {settings.embedding_dimensions}"
        ):
            repository.insert_chunks(
                [StructureAwareChunker().chunk(v1)[0]],
                [[1.0] * (settings.embedding_dimensions - 1)],
            )

        faq_repository.ensure_schema()
        faq = FAQ(
            question=f"Integration question {document_id}",
            answer="The existing FAQ workflow remains available.",
            category="integration",
        )
        faq_repository.upsert(faq, query_embedding)
        faq_matches = faq_repository.search_similar(query_embedding)
        assert any(row["question"] == faq.question for row in faq_matches)
    finally:
        with database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM document_chunks WHERE document_id = %s;",
                    (document_id,),
                )
                cursor.execute(
                    "DELETE FROM faq WHERE question = %s;",
                    (f"Integration question {document_id}",),
                )
