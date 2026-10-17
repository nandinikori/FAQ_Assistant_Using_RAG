"""Markdown block parser that retains line and character source locations."""

from __future__ import annotations

import re

from app.chunking.types import Document, SourceBlock

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?$")


def parse_markdown(document: Document) -> list[SourceBlock]:
    """Parse headings, paragraphs, lists, and tables without losing provenance."""
    lines = document.content.splitlines(keepends=True)
    line_starts: list[int] = []
    position = 0
    for line in lines:
        line_starts.append(position)
        position += len(line)

    blocks: list[SourceBlock] = []
    heading_stack: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index].rstrip("\r\n")
        if not current.strip():
            index += 1
            continue

        heading = _HEADING.match(current)
        if heading:
            level = len(heading.group(1))
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(heading.group(2))
            blocks.append(
                _make_block(
                    document,
                    blocks,
                    "heading",
                    index,
                    index + 1,
                    line_starts[index],
                    line_starts[index] + len(current),
                    tuple(heading_stack),
                    {"level": level},
                )
            )
            index += 1
            continue

        if (
            current.lstrip().startswith("|")
            and index + 1 < len(lines)
            and _TABLE_SEPARATOR.match(lines[index + 1].rstrip("\r\n"))
        ):
            start = index
            table_lines = [current, lines[index + 1].rstrip("\r\n")]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].rstrip("\r\n"))
                index += 1
            headers = _table_cells(table_lines[0])
            table_start = line_starts[start]
            table_end = line_starts[index - 1] + len(lines[index - 1].rstrip("\r\n"))
            row_spans = [
                (line_starts[row_index], line_starts[row_index] + len(lines[row_index].rstrip("\r\n")))
                for row_index in range(start + 2, start + len(table_lines))
            ]
            blocks.append(
                _make_block(
                    document,
                    blocks,
                    "table",
                    start,
                    index,
                    table_start,
                    table_end,
                    tuple(heading_stack),
                    {
                        "headers": headers,
                        "table_header": "\n".join(table_lines[:2]),
                        "header_char_start": table_start,
                        "header_char_end": line_starts[start + 1]
                        + len(lines[start + 1].rstrip("\r\n")),
                        "header_line_start": start + 1,
                        "header_line_end": start + 2,
                        "rows": table_lines[2:],
                        "row_spans": row_spans,
                    },
                )
            )
            continue

        kind = "list" if _LIST_ITEM.match(current) else "paragraph"
        start = index
        while index < len(lines):
            text = lines[index].rstrip("\r\n")
            if not text.strip() or _HEADING.match(text):
                break
            if index > start and text.lstrip().startswith("|"):
                break
            if kind == "list" and not (_LIST_ITEM.match(text) or not text.strip()):
                break
            if kind == "paragraph" and _LIST_ITEM.match(text):
                break
            index += 1
        if index == start:
            index += 1
            continue

        end_char = line_starts[index - 1] + len(lines[index - 1].rstrip("\r\n"))
        blocks.append(
            _make_block(
                document,
                blocks,
                kind,
                start,
                index,
                line_starts[start],
                end_char,
                tuple(heading_stack),
            )
        )

    return blocks


def _make_block(
    document: Document,
    blocks: list[SourceBlock],
    kind: str,
    start_index: int,
    end_index: int,
    char_start: int,
    char_end: int,
    heading_path: tuple[str, ...],
    metadata: dict | None = None,
) -> SourceBlock:
    return SourceBlock(
        block_id=f"{document.document_id}:{document.version}:block-{len(blocks) + 1}",
        kind=kind,
        text=document.content[char_start:char_end],
        line_start=start_index + 1,
        line_end=end_index,
        char_start=char_start,
        char_end=char_end,
        heading_path=heading_path,
        metadata=metadata or {},
    )


def _table_cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]
