"""Parsing and document extraction package."""

from __future__ import annotations

from app.parsing.abstention import ABSTENTION_MARKERS, is_abstention
from app.parsing.clauses import (
    CANONICAL_CLAUSE_TYPES,
    CLAUSE_TYPE_ALIASES,
    clause_type_label,
    normalize_clause_type,
    normalize_clause_types,
)
from app.parsing.extractor import (
    chunk_document,
    extract_and_chunk,
    extract_text_from_pdf,
    extract_text_from_txt,
)

__all__ = [
    "ABSTENTION_MARKERS",
    "CANONICAL_CLAUSE_TYPES",
    "CLAUSE_TYPE_ALIASES",
    "chunk_document",
    "clause_type_label",
    "extract_and_chunk",
    "extract_text_from_pdf",
    "extract_text_from_txt",
    "is_abstention",
    "normalize_clause_type",
    "normalize_clause_types",
]
