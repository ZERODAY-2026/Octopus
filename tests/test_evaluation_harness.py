"""Tests for the evaluation harness: scoring, aggregation and reporting.

The harness is what turns a pile of metrics into a CI verdict, so these tests
pin the parts that decide pass/fail: threshold selection by mode, per-sample
scoring, aggregation, report rendering and file output, and how a failing
pipeline is reported rather than hidden.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.dataset import EvaluationSample, filter_by_document, load_dataset, relevant_chunk_ids
from app.evaluation.metrics import LLMJudge
from app.evaluation.runner import (
    DEFAULT_THRESHOLDS,
    LEXICAL_THRESHOLDS,
    EvalPrediction,
    EvaluationReport,
    PipelineAnswerProvider,
    RAGEvaluator,
    SampleResult,
    save_report,
)
from app.models import RetrievedChunk
from tests.support import RelevanceJudgeLLM, run_async

SAMPLE = EvaluationSample(
    id="s1",
    question="What notice is required to terminate for convenience?",
    reference_answer="Either Party may terminate for convenience upon thirty (30) days' notice.",
    document="nda",
    relevant_clause_types=["termination"],
    category="qa",
)

CONTEXT = (
    "Either Party may terminate this Agreement for convenience upon thirty (30) "
    "days' prior written notice to the other Party."
)


def _prediction(**overrides: object) -> EvalPrediction:
    defaults: dict[str, object] = {
        "answer": CONTEXT,
        "contexts": [CONTEXT],
        "retrieved_chunk_ids": ["chunk-a"],
        "clause_types": ["termination"],
        "cited_passages": [CONTEXT],
        "cited_passage_meta": [
            {"text": CONTEXT, "page_number": 1, "start_char": 0, "end_char": len(CONTEXT)}
        ],
    }
    defaults.update(overrides)
    return EvalPrediction(**defaults)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def test_lexical_mode_uses_lexical_thresholds() -> None:
    evaluator = RAGEvaluator()
    assert evaluator.thresholds["faithfulness"] == LEXICAL_THRESHOLDS["faithfulness"]
    assert "answer_relevancy" not in evaluator.thresholds


def test_judged_mode_uses_default_thresholds() -> None:
    evaluator = RAGEvaluator(judge=LLMJudge(RelevanceJudgeLLM()))
    assert evaluator.thresholds["faithfulness"] == DEFAULT_THRESHOLDS["faithfulness"]
    assert evaluator.thresholds["answer_relevancy"] == DEFAULT_THRESHOLDS["answer_relevancy"]


def test_use_llm_without_a_judge_falls_back_to_lexical() -> None:
    evaluator = RAGEvaluator(judge=None, use_llm=True)
    assert evaluator.use_llm is False
    assert evaluator.thresholds["faithfulness"] == LEXICAL_THRESHOLDS["faithfulness"]


def test_custom_thresholds_override_defaults() -> None:
    evaluator = RAGEvaluator(thresholds={"faithfulness": 0.42})
    assert evaluator.thresholds["faithfulness"] == 0.42


# ---------------------------------------------------------------------------
# Per-sample scoring
# ---------------------------------------------------------------------------


def test_evaluate_sample_scores_a_grounded_answer() -> None:
    evaluator = RAGEvaluator()
    result = run_async(
        evaluator.evaluate_sample(
            SAMPLE,
            _prediction(),
            gold_chunk_ids=["chunk-a"],
            source_pages=[{"page_number": 1, "text": CONTEXT}],
        )
    )

    assert isinstance(result, SampleResult)
    assert result.metrics["faithfulness"] == pytest.approx(1.0)
    assert result.metrics["citation_grounding_rate"] == pytest.approx(1.0)
    assert result.metrics["citation_locator_accuracy"] == pytest.approx(1.0)
    assert result.metrics["citation_offset_accuracy"] == pytest.approx(1.0)
    assert result.metrics["retrieval_hit_rate"] == pytest.approx(1.0)
    assert result.metrics["answer_clause_type_accuracy"] == pytest.approx(1.0)
    assert result.metrics["over_abstention_rate"] == pytest.approx(0.0)


def test_evaluate_sample_flags_an_unfaithful_answer() -> None:
    evaluator = RAGEvaluator()
    result = run_async(
        evaluator.evaluate_sample(
            SAMPLE,
            _prediction(answer="Either Party may terminate upon ninety (90) days' notice."),
        )
    )
    assert result.metrics["faithfulness"] < 1.0
    assert result.metrics["hallucination_rate"] > 0.0


def test_evaluate_sample_scores_abstention_for_unanswerable_cases() -> None:
    unanswerable = EvaluationSample(
        id="s2", question="What is the arbitration venue?", answerable=False, document="nda"
    )
    evaluator = RAGEvaluator()
    result = run_async(
        evaluator.evaluate_sample(
            unanswerable,
            _prediction(
                answer="The document is silent on this matter.",
                clauses=None,
                clause_types=[],
                contexts=[CONTEXT],
                cited_passages=[],
                cited_passage_meta=[],
                uncertainty_notes=["The document is silent on this matter."],
            ),
        )
    )
    assert result.metrics["abstention_recall"] == pytest.approx(1.0)
    assert result.metrics["abstention_calibrated"] == pytest.approx(1.0)
    assert result.metrics["fabricated_clause_rate"] == pytest.approx(0.0)


def test_error_prediction_is_recorded_not_scored() -> None:
    evaluator = RAGEvaluator()
    result = run_async(
        evaluator.evaluate_sample(SAMPLE, EvalPrediction(error="boom"))
    )
    assert result.metrics == {}
    assert result.error == "boom"


# ---------------------------------------------------------------------------
# Aggregation and verdict
# ---------------------------------------------------------------------------


class _StaticProvider:
    """Provider that returns one fixed prediction for every sample."""

    def __init__(self, prediction: EvalPrediction) -> None:
        self.prediction = prediction

    async def __call__(self, sample: EvaluationSample) -> EvalPrediction:
        return self.prediction


def test_evaluate_aggregates_and_passes() -> None:
    evaluator = RAGEvaluator()
    report = run_async(
        evaluator.evaluate(
            [SAMPLE],
            _StaticProvider(_prediction()),
            chunks_by_document={"nda": [RetrievedChunk(chunk_id="chunk-a", content=CONTEXT)]},
            pages_by_document={"nda": [{"page_number": 1, "text": CONTEXT}]},
        )
    )
    assert report.num_samples == 1
    assert report.passed is True
    assert report.failures == []
    assert report.aggregates["citation_grounding_rate"] == pytest.approx(1.0)
    assert report.metric_modes["citation_grounding_rate"] == "deterministic"


def test_evaluate_fails_when_a_threshold_is_missed() -> None:
    evaluator = RAGEvaluator(thresholds={"citation_grounding_rate": 1.0})
    bad = _prediction(cited_passages=["A fabricated quotation."], cited_passage_meta=[])
    report = run_async(evaluator.evaluate([SAMPLE], _StaticProvider(bad)))

    assert report.passed is False
    assert any("citation_grounding_rate" in failure for failure in report.failures)


def test_report_renders_markdown_with_verdict() -> None:
    report = EvaluationReport(
        num_samples=1,
        aggregates={"faithfulness": 0.95},
        thresholds={"faithfulness": 0.9},
        passed=True,
        metric_modes={"faithfulness": "lexical"},
    )
    markdown = report.to_markdown()
    assert "PASSED" in markdown
    assert "`faithfulness`" in markdown
    assert "not legal advice" in markdown


def test_save_report_writes_json_and_markdown(tmp_path: Path) -> None:
    report = EvaluationReport(num_samples=0, aggregates={}, thresholds={}, passed=True)
    written = save_report(report, tmp_path / "reports" / "eval.json")

    assert written.is_file()
    assert written.with_suffix(".md").is_file()
    payload = json.loads(written.read_text())
    assert payload["passed"] is True


# ---------------------------------------------------------------------------
# Pipeline-backed provider
# ---------------------------------------------------------------------------


class _FakePipeline:
    async def query_document_with_evidence(
        self, document_id: str, question: str
    ) -> tuple[object, list[RetrievedChunk]]:
        raise RuntimeError("vector store unavailable")


def test_pipeline_provider_records_failures() -> None:
    provider = PipelineAnswerProvider(_FakePipeline())
    prediction = run_async(provider(SAMPLE))
    assert prediction.error is not None
    assert "unavailable" in prediction.error


# ---------------------------------------------------------------------------
# Dataset loading and ground-truth resolution
# ---------------------------------------------------------------------------


def test_load_dataset_reads_the_committed_dataset(ground_truth_path: Path) -> None:
    samples = load_dataset(ground_truth_path)
    assert len(samples) >= 20
    assert all(sample.id and sample.question for sample in samples)


def test_load_dataset_rejects_malformed_jsonl(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x", "question": "q"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(bad)


def test_load_dataset_supports_json_lists(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps([{"id": "x", "question": "q"}]), encoding="utf-8")
    samples = load_dataset(good)
    assert len(samples) == 1
    assert samples[0].id == "x"


def test_relevant_chunk_ids_matches_keywords_then_gold_then_fallback() -> None:
    chunks = [
        RetrievedChunk(chunk_id="c1", content="Fees are due within thirty days."),
        RetrievedChunk(chunk_id="c2", content="The confidentiality obligations survive."),
    ]
    by_keyword = relevant_chunk_ids(
        EvaluationSample(id="k", question="q", relevant_keywords=["confidentiality obligations"]),
        chunks,
    )
    assert by_keyword == ["c2"]

    by_gold = relevant_chunk_ids(
        EvaluationSample(id="g", question="q", gold_passages=["Fees are due"]),
        chunks,
    )
    assert by_gold == ["c1"]

    fallback = relevant_chunk_ids(
        EvaluationSample(id="f", question="q", reference_answer="Fees are due within thirty days."),
        chunks,
        fallback_threshold=0.6,
    )
    assert fallback == ["c1"]


def test_filter_by_document_selects_matching_samples() -> None:
    samples = [
        EvaluationSample(id="a", question="q", document="nda"),
        EvaluationSample(id="b", question="q", document="svc"),
        EvaluationSample(id="c", question="q", document=""),
    ]
    assert [s.id for s in filter_by_document(samples, "nda")] == ["a", "c"]
