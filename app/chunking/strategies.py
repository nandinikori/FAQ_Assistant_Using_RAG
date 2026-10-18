"""Configurable chunking strategies with stable source provenance."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence

from app.chunking.markdown import parse_markdown
from app.chunking.types import Document, DocumentChunk, SourceBlock

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _tokens(text: str) -> list[re.Match[str]]:
    """Return deterministic lexical token-like units for the offline demo."""
    return list(_TOKEN.finditer(text))


def _token_count(text: str) -> int:
    return len(_tokens(text))


def _chunk(
    document: Document,
    strategy: str,
    text: str,
    char_start: int,
    char_end: int,
    ordinal: int,
    source_blocks: Sequence[SourceBlock],
    *,
    parent_chunk_id: str | None = None,
    metadata: dict | None = None,
    token_count: int | None = None,
) -> DocumentChunk:
    overlapping = [
        block
        for block in source_blocks
        if block.char_start < char_end and block.char_end > char_start
    ]
    path = next(
        (block.heading_path for block in reversed(overlapping) if block.heading_path),
        (),
    )
    chunk_metadata = dict(document.metadata)
    chunk_metadata.update(metadata or {})
    identity = (
        f"{document.document_id}:{document.version}:{strategy}:"
        f"{char_start}:{char_end}:{ordinal}:{parent_chunk_id or ''}:"
        f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}:"
        f"{json.dumps(chunk_metadata, sort_keys=True, default=str)}"
    )
    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    line_start = document.content.count("\n", 0, char_start) + 1
    line_end = document.content.count("\n", 0, char_end) + 1
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document.document_id,
        document_version=document.version,
        source_name=document.source_name,
        strategy=strategy,
        text=text,
        token_count=_token_count(text) if token_count is None else token_count,
        source_char_start=char_start,
        source_char_end=char_end,
        source_line_start=line_start,
        source_line_end=line_end,
        heading_path=path,
        source_block_ids=tuple(block.block_id for block in overlapping),
        parent_chunk_id=parent_chunk_id,
        metadata=chunk_metadata,
    )


def _block_chunk(
    document: Document,
    block: SourceBlock,
    strategy: str,
    ordinal: int,
    source_blocks: Sequence[SourceBlock],
    *,
    text: str | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
    metadata: dict | None = None,
) -> DocumentChunk:
    chunk = _chunk(
        document,
        strategy,
        text if text is not None else block.text,
        block.char_start if char_start is None else char_start,
        block.char_end if char_end is None else char_end,
        ordinal,
        source_blocks,
        metadata=metadata,
    )
    return DocumentChunk(
        **{
            **{field: getattr(chunk, field) for field in chunk.__dataclass_fields__},
            "heading_path": block.heading_path,
        }
    )


def _table_row_chunks(
    document: Document,
    block: SourceBlock,
    max_tokens: int,
    strategy: str,
    ordinal_start: int,
    source_blocks: Sequence[SourceBlock],
) -> list[DocumentChunk]:
    header = str(block.metadata["table_header"])
    header_tokens = _token_count(header)
    rows: list[str] = block.metadata["rows"]
    spans: list[tuple[int, int]] = block.metadata["row_spans"]
    if not rows:
        return [
            _block_chunk(
                document,
                block,
                strategy,
                ordinal_start,
                source_blocks,
                metadata={
                    "table_headers": block.metadata["headers"],
                    "configured_max_tokens": max_tokens,
                },
            )
        ]

    chunks: list[DocumentChunk] = []
    batch_rows: list[int] = []
    batch_tokens = header_tokens
    for row_index, row in enumerate(rows):
        row_tokens = _token_count(row)
        if batch_rows and batch_tokens + row_tokens > max_tokens:
            chunks.append(
                _emit_table_batch(
                document, block, header, rows, spans, batch_rows, max_tokens, strategy,
                    ordinal_start + len(chunks),
                source_blocks,
            )
            )
            batch_rows = []
            batch_tokens = header_tokens
        batch_rows.append(row_index)
        batch_tokens += row_tokens

    if batch_rows:
        chunks.append(
            _emit_table_batch(
                document, block, header, rows, spans, batch_rows, max_tokens, strategy,
                ordinal_start + len(chunks),
                source_blocks,
            )
        )
    return chunks


def _emit_table_batch(
    document: Document,
    block: SourceBlock,
    header: str,
    rows: list[str],
    spans: list[tuple[int, int]],
    indices: list[int],
    max_tokens: int,
    strategy: str,
    ordinal: int,
    source_blocks: Sequence[SourceBlock],
) -> DocumentChunk:
    first, last = indices[0], indices[-1]
    text = "\n".join([header, *(rows[index] for index in indices)])
    return _block_chunk(
        document,
        block,
        strategy,
        ordinal,
        source_blocks,
        text=text,
        char_start=spans[first][0],
        char_end=spans[last][1],
        metadata={
            "table_headers": block.metadata["headers"],
            "table_header_source": {
                "char_start": block.metadata["header_char_start"],
                "char_end": block.metadata["header_char_end"],
                "line_start": block.metadata["header_line_start"],
                "line_end": block.metadata["header_line_end"],
            },
            "table_row_start": first + 1,
            "table_row_end": last + 1,
            "repeated_table_header": True,
            "configured_max_tokens": max_tokens,
        },
    )


class TokenWindowChunker:
    """Fixed overlapping windows using a deterministic token-like tokenizer.

    The included tokenizer is suitable for reproducible offline comparison, not
    exact model billing. Inject a tokenizer returning character-offset spans
    when exact model token boundaries matter.
    """

    def __init__(
        self,
        chunk_size: int = 120,
        overlap: int = 20,
        token_spans: Callable[[str], Sequence[tuple[int, int]]] | None = None,
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and less than chunk_size.")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.token_spans = token_spans

    def chunk(
        self,
        document: Document,
        source_blocks: Sequence[SourceBlock] | None = None,
    ) -> list[DocumentChunk]:
        blocks = parse_markdown(document) if source_blocks is None else source_blocks
        spans = (
            list(self.token_spans(document.content))
            if self.token_spans is not None
            else [(match.start(), match.end()) for match in _tokens(document.content)]
        )
        if any(
            start < 0 or end <= start or end > len(document.content)
            for start, end in spans
        ):
            raise ValueError("token_spans must contain valid character-offset pairs.")
        chunks = []
        step = self.chunk_size - self.overlap
        for ordinal, start in enumerate(range(0, len(spans), step)):
            end = min(start + self.chunk_size, len(spans))
            first, last = spans[start], spans[end - 1]
            chunks.append(
                _chunk(
                    document,
                    "token-window",
                    document.content[first[0]:last[1]],
                    first[0],
                    last[1],
                    ordinal,
                    blocks,
                    metadata={
                        "configured_chunk_size": self.chunk_size,
                        "configured_overlap": self.overlap,
                        "tokenizer": (
                            "injected-offset-tokenizer"
                            if self.token_spans is not None
                            else "unicode-lexical-approximation"
                        ),
                    },
                    token_count=end - start,
                )
            )
            if end == len(spans):
                break
        return chunks


class ParagraphChunker:
    """Group complete paragraphs and sentence boundaries up to a token limit."""

    def __init__(self, max_tokens: int = 160):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        self.max_tokens = max_tokens

    def chunk(
        self,
        document: Document,
        source_blocks: Sequence[SourceBlock] | None = None,
    ) -> list[DocumentChunk]:
        blocks = parse_markdown(document) if source_blocks is None else source_blocks
        chunks: list[DocumentChunk] = []
        pending: list[SourceBlock] = []
        pending_tokens = 0

        def flush() -> None:
            nonlocal pending, pending_tokens
            if not pending:
                return
            first, last = pending[0], pending[-1]
            chunks.append(
                _chunk(
                    document,
                    "paragraph",
                    document.content[first.char_start:last.char_end],
                    first.char_start,
                    last.char_end,
                    len(chunks),
                    blocks,
                    metadata={"configured_max_tokens": self.max_tokens},
                )
            )
            pending, pending_tokens = [], 0

        for block in blocks:
            if block.kind == "table":
                flush()
                chunks.extend(
                    _table_row_chunks(
                        document, block, self.max_tokens, "paragraph", len(chunks),
                        blocks,
                    )
                )
                continue
            pieces = self._split_long_block(block)
            for piece, start, end in pieces:
                piece_tokens = _token_count(piece)
                if pending and (
                    pending_tokens + piece_tokens > self.max_tokens
                    or pending[-1].heading_path != block.heading_path
                ):
                    flush()
                if piece_tokens > self.max_tokens:
                    chunks.append(
                        _chunk(
                            document, "paragraph", piece, start, end, len(chunks), blocks,
                            metadata={
                                "oversized_atomic_sentence": True,
                                "configured_max_tokens": self.max_tokens,
                            },
                        )
                    )
                else:
                    pending.append(
                        SourceBlock(
                            block.block_id, block.kind, piece, block.line_start,
                            block.line_end, start, end, block.heading_path,
                            block.metadata,
                        )
                    )
                    pending_tokens += piece_tokens
        flush()
        return chunks

    def _split_long_block(
        self, block: SourceBlock
    ) -> list[tuple[str, int, int]]:
        if _token_count(block.text) <= self.max_tokens or block.kind == "heading":
            return [(block.text, block.char_start, block.char_end)]
        boundaries = [match.end() for match in _SENTENCE_END.finditer(block.text)]
        ranges = []
        local_start = 0
        for boundary in boundaries:
            sentence_end = boundary
            while sentence_end < len(block.text) and block.text[sentence_end].isspace():
                sentence_end += 1
            ranges.append((local_start, boundary))
            local_start = sentence_end
        if local_start < len(block.text):
            ranges.append((local_start, len(block.text.rstrip())))

        pieces: list[tuple[str, int, int]] = []
        for local_start, local_end in ranges:
            text = block.text[local_start:local_end]
            if _token_count(text) > self.max_tokens:
                tokens = _tokens(text)
                for index in range(0, len(tokens), self.max_tokens):
                    end_index = min(index + self.max_tokens, len(tokens))
                    start_char = local_start + tokens[index].start()
                    end_char = local_start + tokens[end_index - 1].end()
                    pieces.append(
                        (
                            block.text[start_char:end_char],
                            block.char_start + start_char,
                            block.char_start + end_char,
                        )
                    )
            else:
                pieces.append(
                    (
                        text,
                        block.char_start + local_start,
                        block.char_start + local_end,
                    )
                )
        return pieces or [(block.text, block.char_start, block.char_end)]


class RecursiveChunker:
    """Recursively split at preferred text separators, then exact token windows."""

    def __init__(
        self,
        max_tokens: int = 120,
        separators: Sequence[str] = ("\n\n", "\n", " "),
    ):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        if any(separator == "" for separator in separators):
            raise ValueError("separators must be non-empty strings.")
        self.max_tokens = max_tokens
        self.separators = tuple(separators)

    def chunk(
        self,
        document: Document,
        source_blocks: Sequence[SourceBlock] | None = None,
    ) -> list[DocumentChunk]:
        blocks = parse_markdown(document) if source_blocks is None else source_blocks

        def split(start: int, end: int, separator_index: int) -> list[tuple[int, int]]:
            text = document.content[start:end]
            if _token_count(text) <= self.max_tokens:
                return [(start, end)]
            if separator_index >= len(self.separators):
                matches = _tokens(text)
                return [
                    (
                        start + matches[index].start(),
                        start + matches[min(index + self.max_tokens, len(matches)) - 1].end(),
                    )
                    for index in range(0, len(matches), self.max_tokens)
                ]

            separator = self.separators[separator_index]
            boundaries = [
                start + match.end()
                for match in re.finditer(re.escape(separator), text)
            ]
            if not boundaries:
                return split(start, end, separator_index + 1)

            pieces: list[tuple[int, int]] = []
            piece_start = start
            for boundary in [*boundaries, end]:
                if boundary <= piece_start:
                    continue
                if _token_count(document.content[piece_start:boundary]) > self.max_tokens:
                    pieces.extend(split(piece_start, boundary, separator_index + 1))
                else:
                    pieces.append((piece_start, boundary))
                piece_start = boundary
            return pieces

        atomic = split(0, len(document.content), 0)
        packed: list[tuple[int, int]] = []
        for start, end in atomic:
            if not document.content[start:end].strip():
                if packed:
                    packed[-1] = (packed[-1][0], end)
                continue
            if (
                packed
                and _token_count(document.content[packed[-1][0]:end]) <= self.max_tokens
            ):
                packed[-1] = (packed[-1][0], end)
            else:
                packed.append((start, end))

        return [
            _chunk(
                document,
                "recursive",
                document.content[start:end],
                start,
                end,
                ordinal,
                blocks,
                metadata={
                    "configured_max_tokens": self.max_tokens,
                    "separators": list(self.separators),
                    "fallback": "token-offset-splitting",
                },
            )
            for ordinal, (start, end) in enumerate(packed)
        ]


class StructureAwareChunker:
    """Keep heading sections and logical blocks together; repeat table headers."""

    def __init__(self, max_tokens: int = 180):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        self.max_tokens = max_tokens

    def chunk(
        self,
        document: Document,
        source_blocks: Sequence[SourceBlock] | None = None,
    ) -> list[DocumentChunk]:
        blocks = parse_markdown(document) if source_blocks is None else source_blocks
        chunks: list[DocumentChunk] = []
        pending: list[SourceBlock] = []
        pending_tokens = 0

        def flush() -> None:
            nonlocal pending, pending_tokens
            if not pending:
                return
            first, last = pending[0], pending[-1]
            chunks.append(
                _chunk(
                    document,
                    "structure-aware",
                    document.content[first.char_start:last.char_end],
                    first.char_start,
                    last.char_end,
                    len(chunks),
                    blocks,
                    metadata={
                        "block_kinds": [item.kind for item in pending],
                        "configured_max_tokens": self.max_tokens,
                    },
                )
            )
            pending, pending_tokens = [], 0

        for block in blocks:
            if block.kind == "table":
                flush()
                chunks.extend(
                    _table_row_chunks(
                        document,
                        block,
                        self.max_tokens,
                        "structure-aware",
                        len(chunks),
                        blocks,
                    )
                )
                continue
            size = _token_count(block.text)
            if pending and (
                pending_tokens + size > self.max_tokens
                or pending[-1].heading_path != block.heading_path
            ):
                flush()
            if size > self.max_tokens:
                flush()
                chunks.append(
                    _block_chunk(
                        document,
                        block,
                        "structure-aware",
                        len(chunks),
                        blocks,
                        metadata={
                            "oversized_atomic_block": True,
                            "configured_max_tokens": self.max_tokens,
                        },
                    )
                )
            else:
                pending.append(block)
                pending_tokens += size
        flush()
        return chunks


class TableAwareChunker:
    """Group table rows under repeated headers while preserving row locations."""

    def __init__(self, max_tokens: int = 100):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        self.max_tokens = max_tokens

    def chunk(
        self,
        document: Document,
        source_blocks: Sequence[SourceBlock] | None = None,
    ) -> list[DocumentChunk]:
        blocks = parse_markdown(document) if source_blocks is None else source_blocks
        chunks: list[DocumentChunk] = []
        for block in blocks:
            if block.kind == "table":
                chunks.extend(
                    _table_row_chunks(
                        document, block, self.max_tokens, "table-aware", len(chunks),
                        blocks,
                    )
                )
            elif block.kind != "heading":
                chunks.append(
                    _block_chunk(
                        document,
                        block,
                        "table-aware",
                        len(chunks),
                        blocks,
                        metadata={"configured_max_tokens": self.max_tokens},
                    )
                )
        return chunks


class SemanticChunker:
    """Split at low-similarity adjacent-block boundaries using injected scoring."""

    def __init__(
        self,
        similarity: Callable[[str, str], float],
        threshold: float = 0.35,
        max_tokens: int = 180,
    ):
        if not -1 <= threshold <= 1:
            raise ValueError("threshold must be between -1 and 1.")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        self.similarity = similarity
        self.threshold = threshold
        self.max_tokens = max_tokens

    def chunk(
        self,
        document: Document,
        source_blocks: Sequence[SourceBlock] | None = None,
    ) -> list[DocumentChunk]:
        blocks = parse_markdown(document) if source_blocks is None else source_blocks
        chunks: list[DocumentChunk] = []
        pending: list[SourceBlock] = []
        pending_tokens = 0
        for block in blocks:
            if block.kind == "heading":
                if pending:
                    self._flush(document, pending, chunks, blocks)
                    pending, pending_tokens = [], 0
                continue
            if block.kind == "table":
                if pending:
                    self._flush(document, pending, chunks, blocks)
                    pending, pending_tokens = [], 0
                chunks.extend(
                    _table_row_chunks(
                        document, block, self.max_tokens, "semantic", len(chunks),
                        blocks,
                    )
                )
                continue
            block_tokens = _tokens(block.text)
            if len(block_tokens) > self.max_tokens:
                if pending:
                    self._flush(document, pending, chunks, blocks)
                    pending, pending_tokens = [], 0
                for start in range(0, len(block_tokens), self.max_tokens):
                    end = min(start + self.max_tokens, len(block_tokens))
                    first, last = block_tokens[start], block_tokens[end - 1]
                    source_start = block.char_start + first.start()
                    source_end = block.char_start + last.end()
                    chunks.append(
                        _chunk(
                            document,
                            "semantic",
                            document.content[source_start:source_end],
                            source_start,
                            source_end,
                            len(chunks),
                            blocks,
                            metadata={
                                "boundary_signal": "token-limit-fallback",
                                "configured_max_tokens": self.max_tokens,
                            },
                            token_count=end - start,
                        )
                    )
                continue
            should_split = bool(pending) and (
                pending[-1].heading_path != block.heading_path
                or pending_tokens + len(block_tokens) > self.max_tokens
            )
            if should_split:
                self._flush(document, pending, chunks, blocks)
                pending, pending_tokens = [], 0
            if pending:
                similarity = self.similarity(pending[-1].text, block.text)
                if not -1 <= similarity <= 1:
                    raise ValueError("similarity callback must return a score from -1 to 1.")
                if similarity < self.threshold:
                    self._flush(document, pending, chunks, blocks)
                    pending, pending_tokens = [], 0
            pending.append(block)
            pending_tokens += len(block_tokens)
        if pending:
            self._flush(document, pending, chunks, blocks)
        return chunks

    def _flush(
        self,
        document: Document,
        pending: Sequence[SourceBlock],
        chunks: list[DocumentChunk],
        source_blocks: Sequence[SourceBlock],
    ) -> None:
        first, last = pending[0], pending[-1]
        chunks.append(
            _chunk(
                document,
                "semantic",
                document.content[first.char_start:last.char_end],
                first.char_start,
                last.char_end,
                len(chunks),
                source_blocks,
                metadata={
                    "boundary_signal": "injected-similarity-callback",
                    "threshold": self.threshold,
                    "configured_max_tokens": self.max_tokens,
                },
            )
        )


def parent_child_chunks(
    document: Document,
    parent_max_tokens: int = 240,
    child_size: int = 80,
    child_overlap: int = 10,
    source_blocks: Sequence[SourceBlock] | None = None,
) -> list[DocumentChunk]:
    """Build larger contextual parents and overlapping retrieval children."""
    if child_size < 1 or child_overlap < 0 or child_overlap >= child_size:
        raise ValueError("child_size must be positive and exceed child_overlap.")
    if parent_max_tokens < child_size:
        raise ValueError("parent_max_tokens must be >= child_size.")

    blocks = parse_markdown(document) if source_blocks is None else source_blocks
    parents = StructureAwareChunker(parent_max_tokens).chunk(document, blocks)
    results: list[DocumentChunk] = []
    blocks_by_id = {block.block_id: block for block in blocks}
    emitted_tables: set[str] = set()
    for parent_index, parent in enumerate(parents):
        if parent.metadata.get("table_headers"):
            table_block = blocks_by_id[parent.source_block_ids[0]]
            if table_block.block_id in emitted_tables:
                continue
            emitted_tables.add(table_block.block_id)
            parent = _block_chunk(
                document,
                table_block,
                "parent-child-parent",
                parent_index,
                blocks,
                metadata={
                    "role": "parent",
                    "table_headers": table_block.metadata["headers"],
                    "configured_parent_max_tokens": parent_max_tokens,
                },
            )
        else:
            parent = DocumentChunk(
                **{
                    **{field: getattr(parent, field) for field in parent.__dataclass_fields__},
                    "strategy": "parent-child-parent",
                    "metadata": {
                        **parent.metadata,
                        "role": "parent",
                        "configured_parent_max_tokens": parent_max_tokens,
                    },
                }
            )
        results.append(parent)
        text = parent.text
        matches = _tokens(text)
        step = child_size - child_overlap
        for child_index, start in enumerate(range(0, len(matches), step)):
            end = min(start + child_size, len(matches))
            first, last = matches[start], matches[end - 1]
            source_start = parent.source_char_start + first.start()
            source_end = parent.source_char_start + last.end()
            child_text = document.content[source_start:source_end]
            child = _chunk(
                document,
                "parent-child",
                child_text,
                source_start,
                source_end,
                parent_index * 10000 + child_index,
                blocks,
                parent_chunk_id=parent.chunk_id,
                metadata={
                    "role": "child",
                    "configured_child_size": child_size,
                    "configured_child_overlap": child_overlap,
                    "configured_parent_max_tokens": parent_max_tokens,
                },
            )
            results.append(child)
            if end == len(matches):
                break
    return results
