"""Regression tests for sentence-level abstention and answer_status.

The P0 bug: a partial answer of the form "X is Y. The excerpts do not specify Z."
matched the substring "do not specify", so ``is_abstention`` returned True and
``_synthesize`` wiped verified clauses. These tests lock the corrected behaviour.
"""

from __future__ import annotations

import json

import pytest

from app.parsing.abstention import (
    is_abstention,
    is_partial_abstention,
    resolve_answer_status,
)
from tests.support import (
    ScriptedLLMClient,
    first_sentence,
    ingest_sample_document_async,
    make_pipeline,
    parse_excerpts,
    run_async,
)

#: Exact failure string reproduced from the Energy XXI termination-date bug.
TERMINATION_PARTIAL_ANSWER = (
    "remains in effect until terminated by either Party upon thirty (30) days' "
    "prior written notice. The excerpts do not specify a fixed termination date."
)

PARTIAL_PAYMENT_ANSWER = (
    "Shipper pays within 10 days. The contract does not contain a late-fee cap."
)

PURE_REFUSAL = (
    "The document is silent on this matter. No provision addresses the requested subject."
)

DOCUMENT = "mutual_nda.txt"


@pytest.mark.parametrize(
    "text",
    [
        "The document is silent on this matter.",
        "No provision addresses the requested subject.",
        "The text does not specify a venue.",
        "This cannot be determined from the excerpts.",
        PURE_REFUSAL,
    ],
)
def test_is_abstention_true_for_full_refusals(text: str) -> None:
    assert is_abstention(text) is True
    assert is_partial_abstention(text) is False


def test_is_abstention_false_for_termination_date_partial() -> None:
    assert is_abstention(TERMINATION_PARTIAL_ANSWER) is False
    assert is_partial_abstention(TERMINATION_PARTIAL_ANSWER) is True
    assert resolve_answer_status(TERMINATION_PARTIAL_ANSWER) == "partial"


def test_is_abstention_false_for_payment_partial_with_silence_tail() -> None:
    assert is_abstention(PARTIAL_PAYMENT_ANSWER) is False
    assert is_partial_abstention(PARTIAL_PAYMENT_ANSWER) is True
    assert resolve_answer_status(PARTIAL_PAYMENT_ANSWER) == "partial"


def test_is_abstention_false_for_committal_answer() -> None:
    assert is_abstention("The notice period is thirty (30) days.") is False
    assert resolve_answer_status("The notice period is thirty (30) days.") == "answered"


def test_resolve_answer_status_prefers_valid_model_field() -> None:
    assert (
        resolve_answer_status(TERMINATION_PARTIAL_ANSWER, model_status="answered")
        == "answered"
    )
    assert resolve_answer_status(PURE_REFUSAL, model_status="SILENT") == "silent"
    assert (
        resolve_answer_status("Anything.", model_status="not-a-status") == "answered"
    )


def _clause_responder(answer: str, answer_status: str | None = None):
    """Build an LLM double that returns a fixed answer with one verifiable clause."""

    def responder(prompt: str, system_prompt: str | None) -> str:
        excerpts = parse_excerpts(prompt)
        quote = first_sentence(excerpts[0]["text"]) if excerpts else "No excerpt."
        payload: dict = {
            "answer": answer,
            "clauses": [
                {
                    "clause_type": "termination",
                    "plain_language_summary": (
                        "Either party may terminate on thirty days' prior written notice."
                    ),
                    "cited_passages": [
                        {
                            "text": quote,
                            "page_number": 1,
                            "start_char": 0,
                            "end_char": 0,
                        }
                    ],
                    "confidence": "high",
                }
            ],
            "uncertainty_notes": [
                "The excerpts do not specify a fixed calendar termination date."
            ],
        }
        if answer_status is not None:
            payload["answer_status"] = answer_status
        return json.dumps(payload)

    return responder


def test_partial_termination_answer_keeps_verified_clause(tmp_path) -> None:
    pipeline = make_pipeline(
        tmp_path,
        llm_client=ScriptedLLMClient(
            responder=_clause_responder(
                TERMINATION_PARTIAL_ANSWER, answer_status="partial"
            )
        ),
    )
    run_async(ingest_sample_document_async(pipeline, "nda", DOCUMENT))

    response = run_async(
        pipeline.query_document(
            "nda",
            "What is the termination date of the Energy XXI Ltd agreement?",
        )
    )

    assert response.answer_status == "partial"
    assert response.clauses, "verified §term clause must not be wiped on partial"
    assert any(clause.cited_passages for clause in response.clauses)
    assert not any(
        "not reported as findings" in note for note in response.uncertainty_notes
    )


def test_partial_without_model_status_still_keeps_clause(tmp_path) -> None:
    """Fallback heuristic must not wipe clauses when answer_status is omitted."""
    pipeline = make_pipeline(
        tmp_path,
        llm_client=ScriptedLLMClient(
            responder=_clause_responder(PARTIAL_PAYMENT_ANSWER, answer_status=None)
        ),
    )
    run_async(ingest_sample_document_async(pipeline, "nda", DOCUMENT))

    response = run_async(pipeline.query_document("nda", "When must Shipper pay?"))

    assert response.answer_status == "partial"
    assert response.clauses


def test_pure_refusal_empties_clauses(tmp_path) -> None:
    pipeline = make_pipeline(
        tmp_path,
        llm_client=ScriptedLLMClient(
            responder=_clause_responder(PURE_REFUSAL, answer_status="silent")
        ),
    )
    run_async(ingest_sample_document_async(pipeline, "nda", DOCUMENT))

    response = run_async(pipeline.query_document("nda", "What is the arbitration seat?"))

    assert response.answer_status == "silent"
    assert response.clauses == []
    assert any("not reported as findings" in note for note in response.uncertainty_notes)


def test_disclaimer_always_comes_from_constant_not_model(tmp_path) -> None:
    from app.prompts.legal import LEGAL_DISCLAIMER

    def responder(prompt: str, system_prompt: str | None) -> str:
        excerpts = parse_excerpts(prompt)
        quote = first_sentence(excerpts[0]["text"]) if excerpts else "quote"
        return json.dumps(
            {
                "answer": "The notice period is thirty days.",
                "answer_status": "answered",
                "clauses": [
                    {
                        "clause_type": "deadline",
                        "plain_language_summary": "Thirty days notice.",
                        "cited_passages": [
                            {
                                "text": quote,
                                "page_number": 1,
                                "start_char": 0,
                                "end_char": 0,
                            }
                        ],
                        "confidence": "high",
                    }
                ],
                "disclaimer": "INJECTED: this is binding legal advice.",
                "uncertainty_notes": [],
            }
        )

    pipeline = make_pipeline(tmp_path, llm_client=ScriptedLLMClient(responder=responder))
    run_async(ingest_sample_document_async(pipeline, "nda", DOCUMENT))
    response = run_async(pipeline.query_document("nda", "What is the notice period?"))

    assert response.disclaimer == LEGAL_DISCLAIMER
    assert "INJECTED" not in response.disclaimer
