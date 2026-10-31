import json

import pytest

from app.retrieval.evaluation import (
    EvaluationQuestion,
    cosine_similarity,
    load_evaluation_questions,
    run_retrieval_evaluation,
)
from app.chunking.corpus import load_policy_corpus
from app.chunking.types import Document
from app.chunking.strategies import StructureAwareChunker
from app.retrieval.retrieval import bm25_rank, reciprocal_rank_fusion


class DistinctiveEmbeddingClient:
    model = "test-discriminative-model"
    dimensions = 2
    configuration_version = "test-v1"

    def __init__(self):
        self.calls = []

    def generate_embedding(self, text):
        self.calls.append(text)
        lowered = text.casefold()
        if "domestic" in lowered:
            return [1.0, 0.0]
        if "travel" in lowered:
            return [0.0, 1.0]
        return [0.1, 0.1]


def test_cosine_similarity_ranks_distinct_vectors_and_rejects_invalid_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="equal length"):
        cosine_similarity([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


def test_load_evaluation_questions_rejects_duplicate_ids(tmp_path):
    query_file = tmp_path / "queries.json"
    record = {
        "id": "duplicate",
        "question": "What is the rule?",
        "document_id": "policy",
        "document_version": "1",
        "required_evidence": ["the rule"],
    }
    query_file.write_text(json.dumps([record, record]), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate evaluation query id"):
        load_evaluation_questions(query_file)


def test_evaluation_ranks_evidence_and_caches_discriminative_embeddings():
    noise = " ".join(f"unrelated{index}" for index in range(110))
    document = Document(
        "travel",
        "1",
        "travel.md",
        (
            f"# Travel policy\n\n{noise}\n\n"
            "## Reimbursement limits\n\n"
            "| Region | Lodging per night | Meals per day | Receipt threshold |\n"
            "| --- | ---: | ---: | ---: |\n"
            "| Domestic | USD 220 | USD 65 | USD 25 |"
        ),
    )
    question = EvaluationQuestion(
        question_id="domestic-limits",
        question="What is the domestic lodging limit?",
        document_id="travel",
        document_version="1",
        required_evidence=("Domestic | USD 220 | USD 65 | USD 25",),
    )
    embedding_client = DistinctiveEmbeddingClient()

    report = run_retrieval_evaluation(
        [document], [question], embedding_client, top_k=1
    )

    assert len(report["strategies"]) == 7
    table_aware = next(
        result for result in report["strategies"] if result["strategy"] == "table-aware"
    )
    assert table_aware["metrics"]["hit_at_k"] == 1.0
    assert table_aware["metrics"]["evidence_recall_at_k"] == 1.0
    assert (
        table_aware["queries"][0]["evidence_ranks"][
            "Domestic | USD 220 | USD 65 | USD 25"
        ]
        == 1
    )
    assert report["embedding_call_count"] == len(embedding_client.calls)
    assert report["unique_embedding_text_count"] == len(set(embedding_client.calls))
    assert report["embedding"]["vectors_fingerprint_sha256"]
    assert report["corpus_fingerprint_sha256"]
    assert report["query_fingerprint_sha256"]
    assert report["winner"] == (
        f"{report['winner_strategy']} + {report['winner_methodology']}"
    )
    assert {
        method
        for strategy in report["strategies"]
        for method in strategy["methodologies"]
    } == {"dense", "bm25", "rrf", "dense-rerank"}
    assert len(report["ranked_combinations"]) == 28


def test_evaluation_rejects_evidence_missing_from_labeled_source():
    document = Document("policy", "1", "policy.md", "The written rule applies.")
    question = EvaluationQuestion(
        question_id="missing",
        question="What does the rule say?",
        document_id="policy",
        document_version="1",
        required_evidence=("not in source",),
    )

    with pytest.raises(ValueError, match="evidence is absent"):
        run_retrieval_evaluation(
            [document], [question], DistinctiveEmbeddingClient()
        )


def test_parent_child_evaluation_expands_retrieved_child_to_parent_context():
    body = (
        "Domestic "
        + " ".join(f"unrelated{index}" for index in range(170))
        + " distinctive evidence marker."
    )
    document = Document("policy", "1", "policy.md", f"# Policy\n\n{body}")
    question = EvaluationQuestion(
        question_id="parent-context",
        question="What applies to domestic work?",
        document_id="policy",
        document_version="1",
        required_evidence=("distinctive evidence marker",),
    )
    report = run_retrieval_evaluation(
        [document], [question], DistinctiveEmbeddingClient(), top_k=1
    )

    parent_child = next(
        result
        for result in report["strategies"]
        if result["strategy"] == "parent-child"
    )
    top_chunk = parent_child["queries"][0]["top_chunks"][0]
    assert (
        "distinctive evidence marker"
        not in document.content[
            top_chunk["source_char_start"]:top_chunk["source_char_end"]
        ]
    )
    assert (
        parent_child["queries"][0]["evidence_ranks"][
            "distinctive evidence marker"
        ]
        == 1
    )


def test_bundled_labels_match_policy_corpus_and_run_without_database():
    questions = load_evaluation_questions()
    report = run_retrieval_evaluation(
        load_policy_corpus(), questions, DistinctiveEmbeddingClient(), top_k=3
    )

    assert report["question_count"] == 8
    assert report["embedding"]["model"] == "test-discriminative-model"
    assert len(report["strategies"]) == 7


def test_bm25_retrieves_exact_term_and_rrf_fuses_rankings():
    document = Document(
        "policy",
        "1",
        "policy.md",
        "Domestic lodging limit is USD 220.\n\n"
        "International lodging limit is USD 300.",
    )
    chunks = StructureAwareChunker(max_tokens=10).chunk(document)
    rows = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "heading_path": list(chunk.heading_path),
        }
        for chunk in chunks
    ]
    bm25 = bm25_rank("Domestic lodging USD 220", rows, limit=len(rows))

    assert "Domestic" in bm25[0]["text"]
    fused = reciprocal_rank_fusion(bm25, list(reversed(bm25)), limit=len(chunks))
    assert {row["chunk_id"] for row in fused} == {chunk.chunk_id for chunk in chunks}
    assert fused[0]["chunk_id"] in {row["chunk_id"] for row in bm25}
