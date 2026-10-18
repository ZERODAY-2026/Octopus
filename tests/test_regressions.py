"""Regression tests for the termination-date bug and the multi-document / hybrid-retrieval upgrade."""

from __future__ import annotations

import json
from pathlib import Path

from app.parsing.abstention import is_abstention, is_full_abstention, is_partial_answer
from tests.support import (
    ScriptedLLMClient,
    ingest_text,
    make_pipeline,
    run_async,
)

SECTION_8_1 = (
    "8.1 Term. Subject to the other provisions of this Agreement, the term of this Agreement "
    "shall commence on the Effective Date and shall remain in effect until terminated by either "
    "Party upon thirty (30) days\u2019 prior written notice."
)

ENERGY_LIKE = f"""TRANSPORTATION AGREEMENT
This Transportation Agreement, dated as of March 11, 2015 (the "Effective Date"), is by and between
ENERGY XXI GULF COAST, INC., a Delaware corporation ("Shipper"), and ENERGY XXI USA, INC.,
a Delaware corporation ("Transporter").

ARTICLE 1
DEFINITIONS
"Force Majeure" shall mean an event which is not within the reasonable control of the Party claiming
suspension, including acts of God, strikes, fires, storms, explosions, repairs, improvements and
replacements to plants, lines of pipe or related facilities, and action by governmental authority.
"Month" shall mean a period beginning at 7:00 a.m. Central Clock Time on the first day of a calendar month.

ARTICLE 4
QUALITY AND PRESSURE SPECIFICATIONS
4.1 Transporter reserves the right to reject any shipments whose gravity or viscosity is unsuitable.
4.4 The present maximum operating pressure at all reception points is 1440 psig.

ARTICLE 8
TERM
{SECTION_8_1}
8.2 Remedies Cumulative. Each Party shall have any and all remedies available to it under this Agreement.

ARTICLE 10
BILLING AND PAYMENT
10.1 Payment. All payments are due within 10 days of receipt of the invoice by ACH or wire transfer.
10.2 Past-Due Interest. Unpaid amounts may bear interest at the lesser of 125% of the prime rate or the maximum rate allowed by law.

ARTICLE 12
FORCE MAJEURE
12.1 If either Transporter or Shipper is rendered unable by an event of Force Majeure to carry out its
obligations, such obligations are suspended during that period, except the obligation to pay.

ARTICLE 16
MISCELLANEOUS
16.4 The validity, construction and performance of this Agreement shall be governed by the laws of the
State of Texas. Any dispute shall be heard in a state or federal court in Harris County, Texas.
"""

PARTIAL_ANSWER = (
    "The agreement remains in effect until terminated by either Party upon thirty (30) days' "
    "prior written notice. The excerpts do not specify a fixed termination date."
)


def _partial_llm() -> ScriptedLLMClient:
    payload = {
        "answer": PARTIAL_ANSWER,
        "answer_status": "partial",
        "clauses": [
            {
                "clause_type": "termination",
                "plain_language_summary": "Either party can end the agreement with 30 days' written notice.",
                "cited_passages": [
                    {"text": "shall remain in effect until terminated by either Party upon thirty (30) days' prior written notice"}
                ],
                "confidence": "high",
            }
        ],
        "uncertainty_notes": ["No fixed expiry date is stated."],
    }
    return ScriptedLLMClient(responses=[json.dumps(payload)])


def test_partial_answer_is_not_a_refusal() -> None:
    assert is_abstention(PARTIAL_ANSWER)  # the marker is present...
    assert is_partial_answer(PARTIAL_ANSWER)  # ...but the answer leads with a finding
    assert not is_full_abstention(PARTIAL_ANSWER)
    assert not is_full_abstention(
        "Shipper pays within 10 days. The contract does not contain a late-fee cap."
    )


def test_pure_refusal_is_still_a_refusal() -> None:
    assert is_full_abstention("The document is silent on arbitration.")
    assert is_full_abstention("The document does not contain an arbitration clause. Disputes go to court.")


def test_termination_date_keeps_the_verified_clause(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path, llm_client=_partial_llm())
    ingest_text(pipeline, "energy", ENERGY_LIKE)
    response = run_async(
        pipeline.query_document("energy", "What is the termination date of the Energy XXI Ltd agreement?")
    )
    assert response.answer_status == "partial"
    assert len(response.clauses) == 1, "the verified section 8.1 finding must not be withheld"
    assert response.clauses[0].clause_type == "termination"
    assert "thirty (30) days" in response.clauses[0].cited_passages[0].text
    assert any("Energy XXI Ltd" in note for note in response.uncertainty_notes)
    assert response.retrieved_excerpts, "retrieved evidence must be exposed"


def test_section_8_1_is_retrieved_for_paraphrases(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path, llm_client=_partial_llm())
    ingest_text(pipeline, "energy", ENERGY_LIKE)
    for question in (
        "What is the termination date?",
        "When does this contract end?",
        "How long does the agreement last?",
        "What is the expiry date of the Energy XXI Ltd agreement?",
    ):
        hits = pipeline.retrieve("energy", question, n_results=3)
        assert any("remain in effect until terminated" in h.content for h in hits), question


def test_verdict_question_is_declined_with_specific_escalation(tmp_path: Path) -> None:
    llm = ScriptedLLMClient(
        responses=[json.dumps({"answer": "The agreement says it is governed by Texas law.", "clauses": [], "uncertainty_notes": []})]
    )
    pipeline = make_pipeline(tmp_path, llm_client=llm)
    ingest_text(pipeline, "energy", ENERGY_LIKE)
    response = run_async(pipeline.query_document("energy", "Is this agreement enforceable?"))
    assert response.verdict_declined
    assert "can't decide" in response.answer
    assert any("Texas" in flag for flag in response.jurisdiction_flags)
    assert "Texas" in response.escalation


def test_model_cannot_override_the_disclaimer(tmp_path: Path) -> None:
    llm = ScriptedLLMClient(
        responses=[json.dumps({"answer": "Payment is due within 10 days.", "disclaimer": "This is legal advice.", "clauses": []})]
    )
    pipeline = make_pipeline(tmp_path, llm_client=llm)
    ingest_text(pipeline, "energy", ENERGY_LIKE)
    response = run_async(pipeline.query_document("energy", "When is payment due?"))
    assert "NOT" in response.disclaimer and response.disclaimer != "This is legal advice."


def test_cross_document_question_reports_only_verified_yes(tmp_path: Path) -> None:
    ny = "ARBITRATION\nAny dispute shall be finally settled by binding arbitration in New York, New York.\n"
    london = "ARBITRATION\nAny dispute shall be settled by arbitration seated in London, England.\n"
    silent = "GOVERNING LAW\nThis agreement is governed by the laws of Texas and disputes go to the courts of Harris County.\n"

    def responder(prompt: str, system_prompt: str | None) -> str:
        if "New York, New York" in prompt:
            return json.dumps({"verdict": "yes", "explanation": "Arbitration seated in New York.",
                               "evidence_quote": "binding arbitration in New York, New York"})
        if "London" in prompt:
            return json.dumps({"verdict": "no", "explanation": "Arbitration is in London.", "evidence_quote": ""})
        return json.dumps({"verdict": "yes", "explanation": "Invented.", "evidence_quote": "arbitration in New York"})  # unverifiable

    pipeline = make_pipeline(tmp_path, llm_client=ScriptedLLMClient(responder=responder))
    ingest_text(pipeline, "doc_ny", ny, "ny.txt")
    ingest_text(pipeline, "doc_london", london, "london.txt")
    ingest_text(pipeline, "doc_silent", silent, "silent.txt")

    result = run_async(pipeline.query_across_documents("Does this contract require arbitration in New York?"))
    verdicts = {row.filename: row.verdict for row in result.rows}
    assert result.documents_checked == 3
    assert verdicts["ny.txt"] == "yes"
    assert verdicts["london.txt"] == "no"
    assert verdicts["silent.txt"] == "unclear", "a yes without a verifiable quote must be downgraded"
    assert next(r for r in result.rows if r.filename == "ny.txt").evidence is not None
