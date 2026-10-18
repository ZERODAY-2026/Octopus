"""Clause-type normalization and its effect on the pipeline and the metrics.

A local model names clauses with compound and aliased labels ("obligation |
penalty | termination", "term_and_termination", "Limitation of Liability"). Two
guarantees are tested here:

1. Every label is folded onto the canonical vocabulary, and a compound label
   becomes one finding per category — no category is lost inside a pipe.
2. A finding is withheld only when ``answer_status`` is ``silent`` (a full
   refusal). Partial answers that both state a finding and note silence on
   another point must keep their verified clauses.
"""

from __future__ import annotations

import json

import pytest

from app.parsing.clauses import (
    CANONICAL_CLAUSE_TYPES,
    CLAUSE_TYPE_ALIASES,
    canonical_clause_vocabulary,
    clause_type_label,
    is_canonical,
    normalize_clause_type,
    normalize_clause_types,
    slugify_clause_type,
)
from app.pipeline.triage import _build_clause_analyses
from tests.support import (
    ScriptedLLMClient,
    first_sentence,
    ingest_sample_document_async,
    make_pipeline,
    parse_excerpts,
    run_async,
)

DOCUMENT = "mutual_nda.txt"


# ---------------------------------------------------------------------------
# The vocabulary itself
# ---------------------------------------------------------------------------


def test_canonical_vocabulary_is_slugged_and_unique() -> None:
    assert len(set(CANONICAL_CLAUSE_TYPES)) == len(CANONICAL_CLAUSE_TYPES)
    for label in CANONICAL_CLAUSE_TYPES:
        assert label == slugify_clause_type(label)
        assert is_canonical(label)


def test_every_alias_target_is_a_canonical_type() -> None:
    for alias, target in CLAUSE_TYPE_ALIASES.items():
        assert alias == slugify_clause_type(alias), alias
        assert is_canonical(target), f"{alias!r} points at unknown {target!r}"


def test_canonical_vocabulary_returns_a_stable_sequence() -> None:
    assert list(canonical_clause_vocabulary()) == list(CANONICAL_CLAUSE_TYPES)


# ---------------------------------------------------------------------------
# Splitting and aliasing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("obligation | penalty | termination", ["obligation", "penalty", "termination"]),
        ("obligation | penalty", ["obligation", "penalty"]),
        ("deadline, termination", ["deadline", "termination"]),
        ("dispute-resolution / arbitration", ["dispute_resolution"]),
        ("Limitation of Liability", ["liability"]),
        ("term_and_termination", ["termination"]),
        ("indemnity", ["liability"]),
        ("indemnification", ["liability"]),
        ("NDA", ["confidentiality"]),
        ("None", ["other"]),
        ("not_applicable", ["other"]),
        ("   ", []),
        ("", []),
    ],
)
def test_normalize_clause_type_folds_aliases_and_splits_compounds(
    label: str, expected: list[str]
) -> None:
    assert normalize_clause_type(label) == expected


def test_normalize_clause_type_keeps_an_unknown_label_as_its_slug() -> None:
    # The vocabulary is allowed to grow without silently discarding a label the
    # model invented; an unrecognized but well-formed label survives as a slug.
    assert normalize_clause_type("Escrow Arrangement") == ["escrow_arrangement"]
    assert not is_canonical("escrow_arrangement")


def test_normalize_clause_type_deduplicates_within_one_label() -> None:
    assert normalize_clause_type("liability | limitation_of_liability") == ["liability"]


def test_normalize_clause_types_deduplicates_across_labels() -> None:
    assert normalize_clause_types(["liability", "Indemnity", "NDA"]) == [
        "liability",
        "confidentiality",
    ]
    assert normalize_clause_types([]) == []


def test_clause_type_label_renders_and_falls_back() -> None:
    assert clause_type_label("obligation | penalty") == "obligation / penalty"
    assert clause_type_label("   ") == "other"


# ---------------------------------------------------------------------------
# The pipeline emits one finding per canonical category
# ---------------------------------------------------------------------------


def test_build_clause_analyses_splits_compound_labels() -> None:
    clauses = _build_clause_analyses(
        [
            {
                "clause_type": "obligation | penalty | termination",
                "plain_language_summary": "A summary.",
                "cited_passages": [],
                "confidence": "high",
            }
        ]
    )
    assert [clause.clause_type for clause in clauses] == [
        "obligation",
        "penalty",
        "termination",
    ]
    assert {clause.plain_language_summary for clause in clauses} == {"A summary."}
    assert {clause.confidence for clause in clauses} == {"high"}


def test_build_clause_analyses_falls_back_to_other_for_a_blank_label() -> None:
    clauses = _build_clause_analyses([{"clause_type": "", "plain_language_summary": ""}])
    assert [clause.clause_type for clause in clauses] == ["other"]


def _responder(answer: str):
    """Build an LLM double that returns a fixed answer with one compound clause.

    Args:
        answer: The answer text the model will produce.

    Returns:
        A ``responder`` callback for :class:`ScriptedLLMClient`.
    """

    def responder(prompt: str, system_prompt: str | None) -> str:
        excerpts = parse_excerpts(prompt)
        quote = first_sentence(excerpts[0]["text"]) if excerpts else "No excerpt."
        from app.parsing.abstention import resolve_answer_status

        return json.dumps(
            {
                "answer": answer,
                "answer_status": resolve_answer_status(answer),
                "clauses": [
                    {
                        "clause_type": "obligation | penalty",
                        "plain_language_summary": "A reported provision.",
                        "cited_passages": [
                            {
                                "text": quote,
                                "page_number": 1,
                                "start_char": 0,
                                "end_char": 0,
                            }
                        ],
                        "confidence": "medium",
                    }
                ],
                "uncertainty_notes": ["The document is silent on this matter."],
            }
        )

    return responder


def test_committal_answer_reports_one_clause_per_canonical_type(tmp_path) -> None:
    answer = "The parties owe each other a duty of confidence and a penalty applies."
    pipeline = make_pipeline(tmp_path, llm_client=ScriptedLLMClient(responder=_responder(answer)))
    run_async(ingest_sample_document_async(pipeline, "nda", DOCUMENT))

    response = run_async(pipeline.query_document("nda", "What penalties apply?"))

    assert sorted(clause.clause_type for clause in response.clauses) == [
        "obligation",
        "penalty",
    ]


def test_a_refusal_withholds_nearby_clause_findings(tmp_path) -> None:
    answer = "The document is silent on the exact monetary penalty for disclosure."
    pipeline = make_pipeline(tmp_path, llm_client=ScriptedLLMClient(responder=_responder(answer)))
    run_async(ingest_sample_document_async(pipeline, "nda", DOCUMENT))

    response = run_async(pipeline.query_document("nda", "What is the monetary penalty?"))

    assert response.answer_status == "silent"
    assert response.clauses == []
    assert any("not reported as findings" in note for note in response.uncertainty_notes)


def test_partial_answer_keeps_nearby_verified_clause_findings(tmp_path) -> None:
    answer = (
        "Shipper pays within 10 days. The contract does not contain a late-fee cap."
    )
    pipeline = make_pipeline(tmp_path, llm_client=ScriptedLLMClient(responder=_responder(answer)))
    run_async(ingest_sample_document_async(pipeline, "nda", DOCUMENT))

    response = run_async(pipeline.query_document("nda", "What penalties apply?"))

    assert response.answer_status == "partial"
    assert sorted(clause.clause_type for clause in response.clauses) == [
        "obligation",
        "penalty",
    ]
