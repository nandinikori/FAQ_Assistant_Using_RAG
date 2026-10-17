"""Load the versioned synthetic policy corpus used by the chunking demo."""

from __future__ import annotations

import json
from pathlib import Path

from app.chunking.types import Document

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data" / "policies"


def load_policy_corpus(corpus_dir: str | Path = DEFAULT_CORPUS) -> list[Document]:
    root = Path(corpus_dir)
    manifest_path = root / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    documents = []
    for record in manifest["documents"]:
        source_path = (root / record["source_path"]).resolve()
        if root.resolve() not in source_path.parents:
            raise ValueError(f"Policy source path escapes corpus directory: {record['source_path']}")
        content = source_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                document_id=record["document_id"],
                version=record["version"],
                source_name=record["source_path"],
                content=content,
                metadata={
                    **record.get("metadata", {}),
                    "source_path": record["source_path"],
                },
            )
        )
    return documents
