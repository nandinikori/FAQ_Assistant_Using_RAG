import re

import pytest

from app.chunking.corpus import load_policy_corpus
from app.chunking.evaluation import evaluate_chunks
from app.chunking.markdown import parse_markdown
from app.chunking.strategies import (
    ParagraphChunker,
    RecursiveChunker,
    SemanticChunker,
    StructureAwareChunker,
    TableAwareChunker,
    TokenWindowChunker,
    parent_child_chunks,
)
from app.chunking.types import Document


def test_policy_corpus_preserves_versions_and_synthetic_metadata():
    documents = load_policy_corpus()

    assert [(document.document_id, document.version) for document in documents] == [
        ("remote-work", "1.0"),
        ("remote-work", "2.0"),
        ("travel-expense", "1.0"),
    ]
    assert all(document.metadata["document_type"] == "synthetic-policy" for document in documents)
    assert "Weekly remote days" in documents[0].content
    assert "Weekly remote days" in documents[1].content


def test_markdown_parser_retains_heading_table_and_exact_source_locations():
    document = Document(
        document_id="test-policy",
        version="1",
        source_name="policy.md",
        content="# Policy\n\n## Limits\n\n| Type | Amount |\n| --- | ---: |\n| Meal | 50 |\n",
    )

    blocks = parse_markdown(document)
    table = next(block for block in blocks if block.kind == "table")

    assert table.heading_path == ("Policy", "Limits")
    assert table.metadata["headers"] == ["Type", "Amount"]
    assert table.text == "| Type | Amount |\n| --- | ---: |\n| Meal | 50 |"
    assert document.content[table.char_start:table.char_end] == table.text
    assert table.line_start == 5
    assert table.line_end == 7


def test_token_window_honors_window_overlap_and_can_use_injected_offsets():
    document = Document(
        document_id="tokens",
        version="1",
        source_name="tokens.md",
        content="one two three four five",
    )
    chunker = TokenWindowChunker(
        chunk_size=3,
        overlap=1,
        token_spans=lambda text: [(match.start(), match.end()) for match in re.finditer(r"\S+", text)],
        tokenizer_id="whitespace-v1",
    )

    chunks = chunker.chunk(document)

    assert [chunk.text for chunk in chunks] == ["one two three", "three four five"]
    assert all(chunk.metadata["tokenizer"] == "whitespace-v1" for chunk in chunks)


def test_custom_tokenizer_with_captured_configuration_requires_stable_identity():
    def tokenizer_for(delimiter):
        def tokenize(text):
            return [(match.start(), match.end()) for match in re.finditer(delimiter, text)]

        return tokenize

    comma_tokenizer = tokenizer_for(",")
    space_tokenizer = tokenizer_for(" ")

    with pytest.raises(ValueError, match="explicit stable identity"):
        TokenWindowChunker(token_spans=comma_tokenizer)

    assert TokenWindowChunker(
        token_spans=comma_tokenizer, tokenizer_id="comma-v1"
    ).tokenizer_id != TokenWindowChunker(
        token_spans=space_tokenizer, tokenizer_id="space-v1"
    ).tokenizer_id


def test_table_aware_chunking_repeats_headers_and_keeps_row_provenance():
    document = next(doc for doc in load_policy_corpus() if doc.document_id == "travel-expense")
    chunks = TableAwareChunker(max_tokens=24).chunk(document)
    table_chunks = [chunk for chunk in chunks if chunk.metadata.get("table_headers")]

    assert len(table_chunks) > 1
    assert all("| Region | Lodging per night | Meals per day | Receipt threshold |" in chunk.text for chunk in table_chunks)
    assert all(chunk.metadata["repeated_table_header"] for chunk in table_chunks)
    assert all(
        document.content[
            chunk.metadata["table_header_source"]["char_start"]:
            chunk.metadata["table_header_source"]["char_end"]
        ].startswith("| Region |")
        for chunk in table_chunks
    )
    assert all(chunk.source_line_start <= chunk.source_line_end for chunk in table_chunks)


def test_structure_and_paragraph_chunkers_preserve_source_offsets():
    document = next(doc for doc in load_policy_corpus() if doc.version == "2.0")
    for chunks in (
        StructureAwareChunker(max_tokens=80).chunk(document),
        ParagraphChunker(max_tokens=80).chunk(document),
    ):
        assert chunks
        for chunk in chunks:
            assert document.content[chunk.source_char_start:chunk.source_char_end]
            assert chunk.document_version == "2.0"
            assert chunk.source_block_ids


def test_recursive_chunker_respects_token_limit_and_source_text():
    document = Document(
        document_id="recursive",
        version="1",
        source_name="recursive.md",
        content="# Heading\n\n" + ("alpha beta gamma delta. " * 30),
    )
    chunks = RecursiveChunker(max_tokens=12).chunk(document)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 12 for chunk in chunks)
    assert all(
        document.content[chunk.source_char_start:chunk.source_char_end] == chunk.text
        for chunk in chunks
    )


def test_semantic_chunker_splits_on_injected_low_similarity():
    document = Document(
        document_id="semantic",
        version="1",
        source_name="semantic.md",
        content="# Policy\n\n## Section\n\nAlpha topic applies.\n\nBeta topic differs.\n",
    )
    def always_different(left, right):
        return 0.0 * (len(left) + len(right))

    chunker = SemanticChunker(
        similarity=always_different,
        threshold=0.5,
        max_tokens=100,
        similarity_id="test-always-different-v1",
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 2
    assert "Alpha topic applies." in chunks[0].text
    assert "Beta topic differs." in chunks[1].text


def test_semantic_chunker_falls_back_to_token_boundaries_for_oversized_blocks():
    document = Document("long", "1", "long.md", " ".join(f"word{index}" for index in range(24)))
    chunks = SemanticChunker(
        lambda left, right: 1.0 + 0.0 * (len(left) + len(right)),
        max_tokens=6,
        similarity_id="test-constant-similarity-v1",
    ).chunk(document)

    assert chunks
    assert all(chunk.token_count <= 6 for chunk in chunks)
    assert all(
        chunk.metadata["boundary_signal"] == "token-limit-fallback"
        for chunk in chunks
    )


def test_parent_child_chunks_link_children_to_parent_and_keep_version():
    document = next(doc for doc in load_policy_corpus() if doc.document_id == "remote-work")
    chunks = parent_child_chunks(document, parent_max_tokens=100, child_size=30, child_overlap=5)
    parents = {chunk.chunk_id: chunk for chunk in chunks if chunk.metadata["role"] == "parent"}
    children = [chunk for chunk in chunks if chunk.metadata["role"] == "child"]

    assert children
    assert all(child.parent_chunk_id in parents for child in children)
    assert all(child.document_version == document.version for child in children)
    assert all(
        document.content[child.source_char_start:child.source_char_end] == child.text
        for child in children
    )


def test_evaluation_reports_coverage_and_size_without_claiming_relevance():
    document = Document("eval", "1", "eval.md", "# Header\n\nContent here.")
    chunks = StructureAwareChunker().chunk(document)

    report = evaluate_chunks(document, chunks, target_max_tokens=50)

    assert report["chunk_count"] == len(chunks)
    assert report["source_coverage_percent"] > 0
    assert report["chunks_over_target"] == 0
