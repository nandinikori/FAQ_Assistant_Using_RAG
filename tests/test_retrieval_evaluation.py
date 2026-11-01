import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.chunking.configuration import (
    canonical_chunking_configuration,
    chunking_configuration_fingerprint,
    metadata_configuration_filter,
)
from app.chunking.corpus import load_policy_corpus
from app.chunking.types import DocumentChunk
from app.retrieval.evaluation import (
    EVIDENCE_SPAN_COVERAGE,
    EvaluationQuestion,
    EvidenceUnit,
    _evaluate_case,
    _permitted_sources,
    load_evaluation_dataset,
    run_retrieval_evaluation,
    write_evaluation_report,
)


class DistinctiveEmbeddingClient:
    model = "test-discriminative-model"
    dimensions = 2
    configuration_version = "test-v1"

    def __init__(self):
        self.calls = []
        self.last_usage_metadata = None

    def generate_embedding(self, text):
        self.calls.append(text)
        self.last_usage_metadata = {
            "prompt_token_count": 1,
            "total_token_count": 2,
        }
        lowered = text.casefold()
        if "travel" in lowered or "tier" in lowered:
            return [0.0, 1.0]
        return [1.0, 0.0]


def _chunk(
    chunk_id,
    document_id="policy",
    version="1",
    line_start=1,
    line_end=1,
    parent_chunk_id=None,
    token_count=4,
    char_start=0,
    char_end=10,
):
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_version=version,
        source_name=f"{document_id}.md",
        strategy="structure-aware",
        text=f"evidence passage {chunk_id}",
        token_count=token_count,
        source_char_start=char_start,
        source_char_end=char_end,
        source_line_start=line_start,
        source_line_end=line_end,
        heading_path=(),
        source_block_ids=(),
        parent_chunk_id=parent_chunk_id,
    )


def _question(evidence, *, answerability="answerable", scope=None):
    return EvaluationQuestion(
        question_id="q1",
        group_id="g1",
        split="dev",
        category="multiple_passages",
        question="What is required?",
        answerability=answerability,
        reference_answer="The required evidence.",
        expected_refusal=None,
        retrieval_scope=scope or {},
        evidence=tuple(evidence),
        review_status="reviewed",
    )


def test_coordinate_metrics_deduplicate_overlapping_chunks_and_require_source_version():
    evidence = (
        EvidenceUnit("e1", "policy", "1", "policy.md", 2, 3, 20, 40, "first passage"),
        EvidenceUnit("e2", "policy", "1", "policy.md", 6, 6, 60, 70, "second passage"),
    )
    question = _question(evidence)
    ranked = [
        (1.0, _chunk("wrong-version", version="2", line_start=2, line_end=3, char_start=20, char_end=40)),
        (0.95, _chunk("tiny-overlap", line_start=3, line_end=3, char_start=20, char_end=29)),
        (0.9, _chunk("first-half", line_start=3, line_end=3, char_start=30, char_end=40)),
        (0.8, _chunk("overlap-duplicate", line_start=3, line_end=3, char_start=30, char_end=40)),
        (0.7, _chunk("second", line_start=6, line_end=6, char_start=60, char_end=70)),
    ]

    result = _evaluate_case(
        question=question,
        ranking=ranked,
        parents={},
        candidate_pool_size=5,
        candidate_limit=10,
        truncated=False,
        mode="within-document",
        permitted={("policy", "1")},
        context_budget_tokens=100,
        success_k=3,
        query_embedding_seconds=0.1,
        retrieval_seconds=0.2,
        reranking_seconds=0.0,
    )

    assert EVIDENCE_SPAN_COVERAGE == 0.5
    assert result["evidence_ranks"] == {"e1": 3, "e2": 5}
    assert result["metrics"]["evidence_recall_at_5"] == 1.0
    assert result["metrics"]["evidence_recall_at_10"] == 1.0
    assert result["metrics"]["mrr_at_10"] == pytest.approx(1 / 3)
    assert result["metrics"]["all_required_evidence_success_at_3"] == 0
    assert result["top_chunks"][1]["matched_evidence_ids"] == []
    assert result["top_chunks"][2]["matched_evidence_ids"] == ["e1"]
    assert result["timing_seconds"]["total_measured_query"] == 0.3


def test_empty_results_and_unanswerable_cases_are_not_positive_evidence_metrics():
    answerable = _question(
        (EvidenceUnit("e1", "policy", "1", "policy.md", 4, 4, 30, 40, "evidence"),)
    )
    empty = _evaluate_case(
        question=answerable,
        ranking=[],
        parents={},
        candidate_pool_size=0,
        candidate_limit=10,
        truncated=False,
        mode="within-document",
        permitted={("policy", "1")},
        context_budget_tokens=20,
        success_k=5,
        query_embedding_seconds=0,
        retrieval_seconds=0,
        reranking_seconds=0,
    )
    assert empty["metrics"]["evidence_recall_at_5"] == 0.0
    assert empty["metrics"]["mrr_at_10"] == 0.0
    assert empty["metrics"]["all_required_evidence_success_at_5"] == 0

    unanswerable = _question((), answerability="unanswerable")
    unanswerable_result = _evaluate_case(
        question=unanswerable,
        ranking=[],
        parents={},
        candidate_pool_size=0,
        candidate_limit=10,
        truncated=False,
        mode="within-document",
        permitted={("policy", "1")},
        context_budget_tokens=20,
        success_k=5,
        query_embedding_seconds=0,
        retrieval_seconds=0,
        reranking_seconds=0,
    )
    assert unanswerable_result["metrics"] is None


def test_parent_child_uses_expanded_source_and_obeys_shared_context_budget():
    parent = _chunk("parent", line_start=1, line_end=5, token_count=8)
    child = _chunk(
        "child",
        line_start=4,
        line_end=4,
        parent_chunk_id="parent",
        token_count=3,
        char_start=6,
        char_end=10,
    )
    question = _question(
        (EvidenceUnit("e1", "policy", "1", "policy.md", 2, 2, 2, 4, "parent evidence"),)
    )
    result = _evaluate_case(
        question=question,
        ranking=[(1.0, child), (0.9, child)],
        parents={"parent": parent},
        candidate_pool_size=1,
        candidate_limit=10,
        truncated=False,
        mode="within-document",
        permitted={("policy", "1")},
        context_budget_tokens=8,
        success_k=5,
        query_embedding_seconds=0,
        retrieval_seconds=0,
        reranking_seconds=0,
    )
    assert result["evidence_ranks"]["e1"] == 1
    assert result["child_evidence_ranks"]["e1"] is None
    assert result["context_evidence_ranks"]["e1"] == 1
    assert result["context"]["expanded_context_tokens"] == 8
    assert result["context"]["expanded_context_chunk_count"] == 1

    over_budget = _evaluate_case(
        question=question,
        ranking=[(1.0, child)],
        parents={"parent": parent},
        candidate_pool_size=1,
        candidate_limit=10,
        truncated=False,
        mode="within-document",
        permitted={("policy", "1")},
        context_budget_tokens=7,
        success_k=5,
        query_embedding_seconds=0,
        retrieval_seconds=0,
        reranking_seconds=0,
    )
    assert over_budget["evidence_ranks"]["e1"] == 1
    assert over_budget["context_evidence_ranks"]["e1"] is None
    assert over_budget["context_metrics"]["evidence_recall_at_5"] == 0.0


def test_repeated_table_header_provenance_matches_header_and_row_evidence():
    chunk = _chunk(
        "table-row",
        line_start=19,
        line_end=19,
        char_start=40,
        char_end=50,
    )
    chunk = DocumentChunk(
        **{
            **{
                field: getattr(chunk, field)
                for field in chunk.__dataclass_fields__
            },
            "metadata": {
                "table_header_source": {
                    "line_start": 15,
                    "line_end": 15,
                    "char_start": 10,
                    "char_end": 20,
                }
            },
        }
    )
    question = _question(
        (
            EvidenceUnit("header", "policy", "1", "policy.md", 15, 15, 10, 20, "table header"),
            EvidenceUnit("row", "policy", "1", "policy.md", 19, 19, 40, 50, "table row"),
        )
    )
    result = _evaluate_case(
        question=question,
        ranking=[(1.0, chunk)],
        parents={},
        candidate_pool_size=1,
        candidate_limit=10,
        truncated=False,
        mode="within-document",
        permitted={("policy", "1")},
        context_budget_tokens=20,
        success_k=5,
        query_embedding_seconds=0,
        retrieval_seconds=0,
        reranking_seconds=0,
    )
    assert result["evidence_ranks"] == {"header": 1, "row": 1}
    assert result["metrics"]["all_required_evidence_success_at_5"] == 1


def test_canonical_chunking_configuration_is_stable_and_distinguishes_parameters():
    first = {
        "chunk_size": 80,
        "overlap": 12,
        "tokenizer": "unicode-lexical-approximation",
    }
    reordered = {
        "tokenizer": "unicode-lexical-approximation",
        "overlap": 12,
        "chunk_size": 80,
    }
    changed = {
        "chunk_size": 80,
        "overlap": 10,
        "tokenizer": "unicode-lexical-approximation",
    }

    assert canonical_chunking_configuration("token-window", first) == (
        canonical_chunking_configuration("token-window", reordered)
    )
    assert chunking_configuration_fingerprint("token-window", first) == (
        chunking_configuration_fingerprint("token-window", reordered)
    )
    assert chunking_configuration_fingerprint("token-window", first) != (
        chunking_configuration_fingerprint("token-window", changed)
    )
    assert chunking_configuration_fingerprint(
        "token-window",
        {**first, "tokenizer": "another-tokenizer"},
    ) != chunking_configuration_fingerprint("token-window", first)
    assert metadata_configuration_filter("token-window", first) == {
        "configured_chunk_size": 80,
        "configured_overlap": 12,
        "tokenizer": "unicode-lexical-approximation",
    }
    with pytest.raises(ValueError, match="Incomplete recursive"):
        canonical_chunking_configuration("recursive", {"max_tokens": 80})


def test_dataset_has_frozen_grouped_splits_pending_labels_and_source_validated_evidence():
    dataset = load_evaluation_dataset()
    assert len(dataset.questions) == 36
    assert len({question.group_id for question in dataset.questions}) == 18
    assert sum(question.split == "dev" for question in dataset.questions) == 18
    assert sum(question.split == "held_out" for question in dataset.questions) == 18
    assert sum(question.answerability == "unanswerable" for question in dataset.questions) == 8
    assert all(question.review_status == "pending_review" for question in dataset.questions)
    assert dataset.sha256


def test_dataset_rejects_split_manifest_drift_and_bad_evidence_coordinates(tmp_path):
    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "policies"
        / "retrieval_eval_queries.json"
    )
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    payload["groups"][0]["split"] = "held_out"
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate or unstable split"):
        load_evaluation_dataset(edited)

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    payload["groups"][0]["gold_evidence"][0]["line_start"] = 1
    payload["groups"][0]["gold_evidence"][0]["line_end"] = 1
    edited.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the cited source"):
        load_evaluation_dataset(edited)


def test_corpus_wide_scope_does_not_use_gold_document_as_a_filter():
    dataset = load_evaluation_dataset()
    question = next(
        item for item in dataset.questions if item.question_id == "remote-v2-days-a"
    )
    documents = {
        (document.document_id, document.version): document
        for document in load_policy_corpus()
    }
    permitted = _permitted_sources(question, "corpus-wide", documents)

    assert permitted == {
        ("remote-work", "1.0"),
        ("remote-work", "2.0"),
        ("travel-expense", "1.0"),
    }
    assert ("travel-expense", "1.0") in permitted


def test_mode_scopes_are_separate_and_group_variants_stay_together():
    dataset = load_evaluation_dataset()
    documents = {
        (document.document_id, document.version): document
        for document in load_policy_corpus()
    }
    version_question = next(
        item for item in dataset.questions if item.question_id == "remote-version-days-a"
    )
    assert _permitted_sources(version_question, "within-document", documents) is None
    assert _permitted_sources(version_question, "corpus-wide", documents) == {
        ("remote-work", "1.0"),
        ("remote-work", "2.0"),
    }
    grouped = [q for q in dataset.questions if q.group_id == version_question.group_id]
    assert len({q.split for q in grouped}) == 1


def test_pending_labels_are_rejected_for_final_evaluation():
    dataset = load_evaluation_dataset()
    with pytest.raises(ValueError, match="requires reviewed labels"):
        run_retrieval_evaluation(
            load_policy_corpus(),
            dataset.questions[:1],
            DistinctiveEmbeddingClient(),
            split="dev",
        )


def test_held_out_evaluation_requires_confirmation_and_reviewed_labels():
    dataset = load_evaluation_dataset()
    held_out_question = next(
        question for question in dataset.questions if question.split == "held_out"
    )
    documents = load_policy_corpus()

    with pytest.raises(ValueError, match="explicit final-release confirmation"):
        run_retrieval_evaluation(
            documents,
            [held_out_question],
            DistinctiveEmbeddingClient(),
            split="held_out",
            allow_pending=True,
        )

    with pytest.raises(ValueError, match="never allows pending labels"):
        run_retrieval_evaluation(
            documents,
            [held_out_question],
            DistinctiveEmbeddingClient(),
            split="held_out",
            allow_pending=True,
            confirm_held_out=True,
        )


def test_local_evaluation_reports_all_metrics_scope_and_machine_readable_files(tmp_path):
    dataset = load_evaluation_dataset()
    questions = (
        next(q for q in dataset.questions if q.question_id == "remote-v2-days-a"),
        next(q for q in dataset.questions if q.question_id == "remote-v2-exception-a"),
        replace(
            next(q for q in dataset.questions if q.answerability == "unanswerable"),
            split="dev",
        ),
    )
    provider = DistinctiveEmbeddingClient()
    report = run_retrieval_evaluation(
        load_policy_corpus(),
        questions,
        provider,
        allow_pending=True,
        split="dev",
        mode="corpus-wide",
        dataset_sha256=dataset.sha256,
    )

    assert len(report["strategies"]) == 3
    assert {
        method
        for strategy in report["strategies"]
        for method in strategy["methodologies"]
    } == {"dense", "bm25", "rrf", "rerank"}
    dense = report["strategies"][0]["methodologies"]["dense"]
    assert dense["sample_count"] == 2
    assert dense["unanswerable_count"] == 1
    assert "evidence_recall_at_5" in dense["metrics"]
    assert "evidence_recall_at_10" in dense["metrics"]
    assert "mrr_at_10" in dense["metrics"]
    assert "all_required_evidence_success_at_5" in dense["metrics"]
    assert any(
        case["searched_document_versions"]
        == [
            {"document_id": "remote-work", "version": "1.0"},
            {"document_id": "remote-work", "version": "2.0"},
            {"document_id": "travel-expense", "version": "1.0"},
        ]
        for case in dense["queries"]
    )
    assert report["embedding_measurements"]["actual_embedding_calls"] == len(
        provider.calls
    )
    assert report["embedding_measurements"]["provider_token_usage"] == {
        "prompt_token_count": len(provider.calls),
        "total_token_count": 2 * len(provider.calls),
    }
    assert report["embedding_measurements"]["corpus_embedding_preparation_seconds"] >= 0
    assert "precomputed corpus vectors" in report["timing_boundaries"]["retrieval"]
    assert "corpus embedding preparation" in report["timing_boundaries"]["total"]
    json_path, markdown_path = write_evaluation_report(report, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == 2
    markdown_report = markdown_path.read_text(encoding="utf-8")
    assert "Aggregate results" in markdown_report
    assert "| N/A |" in markdown_report


def test_within_document_mode_excludes_version_comparison_cases():
    dataset = load_evaluation_dataset()
    questions = tuple(q for q in dataset.questions if q.split == "dev")
    report = run_retrieval_evaluation(
        load_policy_corpus(),
        questions,
        DistinctiveEmbeddingClient(),
        allow_pending=True,
        split="dev",
        mode="within-document",
    )
    assert report["excluded_mode_incompatible_case_count"] == 4
    assert all(
        len(case["searched_document_versions"]) == 1
        for strategy in report["strategies"]
        for method in strategy["methodologies"].values()
        for case in method["queries"]
    )


def test_optional_broader_comparison_preserves_all_existing_chunkers():
    dataset = load_evaluation_dataset()
    question = next(
        q for q in dataset.questions if q.question_id == "remote-v2-days-a"
    )
    report = run_retrieval_evaluation(
        load_policy_corpus(),
        [question],
        DistinctiveEmbeddingClient(),
        allow_pending=True,
        split="dev",
        include_all_chunkers=True,
    )

    assert {item["strategy"] for item in report["strategies"]} == {
        "token-window",
        "recursive",
        "structure-aware",
        "paragraph",
        "table-aware",
        "semantic",
        "parent-child",
    }
    parent_child = next(
        item for item in report["strategies"] if item["strategy"] == "parent-child"
    )
    dense = parent_child["methodologies"]["dense"]
    assert dense["child_ranking_metrics"]["sample_count"] == 1
    assert dense["context_metrics_after_budget"]["sample_count"] == 1
