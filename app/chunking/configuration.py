"""Canonical identities for reproducible chunking configurations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

_REQUIRED_PARAMETERS = {
    "token-window": {"chunk_size", "overlap", "tokenizer"},
    "recursive": {"max_tokens", "separators"},
    "structure-aware": {"max_tokens"},
    "paragraph": {"max_tokens"},
    "table-aware": {"max_tokens"},
    "semantic": {"max_tokens", "threshold", "similarity_id"},
    "parent-child": {"parent_max_tokens", "child_size", "child_overlap"},
}


def validate_chunking_configuration(
    strategy: str, parameters: Mapping[str, Any]
) -> None:
    required = _REQUIRED_PARAMETERS.get(strategy)
    if required is None:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    missing = required.difference(parameters)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(
            f"Incomplete {strategy} chunking configuration; missing: {names}."
        )


def canonical_chunking_configuration(
    strategy: str, parameters: Mapping[str, Any]
) -> str:
    """Serialize a complete chunker configuration deterministically."""
    validate_chunking_configuration(strategy, parameters)
    return json.dumps(
        {"strategy": strategy, "parameters": parameters},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def chunking_configuration_fingerprint(
    strategy: str, parameters: Mapping[str, Any]
) -> str:
    canonical = canonical_chunking_configuration(strategy, parameters)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def metadata_configuration_filter(
    strategy: str, parameters: Mapping[str, Any]
) -> dict[str, Any]:
    """Map canonical parameters to the names persisted in chunk metadata."""
    validate_chunking_configuration(strategy, parameters)
    result: dict[str, Any] = {}
    for key, value in parameters.items():
        metadata_key = (
            key
            if key in {"separators", "threshold", "tokenizer", "similarity_id"}
            else f"configured_{key}"
        )
        result[metadata_key] = value
    return result
