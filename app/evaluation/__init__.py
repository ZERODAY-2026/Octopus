"""Offline RAG evaluation package for the Contracter legal triage agent.

Public surface:

* :mod:`app.evaluation.metrics` — the metric implementations (RAGAS-style
  faithfulness, answer relevancy, context relevance/precision/recall, answer
  correctness, retrieval metrics, and citation verification).
* :mod:`app.evaluation.dataset` — CUAD-style ground-truth sample models and loaders.
* :mod:`app.evaluation.runner` — the harness that runs a dataset through the
  pipeline and produces a pass/fail report.
* :mod:`app.evaluation.text` — deterministic text utilities shared by the metrics.
"""

from __future__ import annotations

from app.evaluation.dataset import EvaluationSample, load_dataset, relevant_chunk_ids
from app.evaluation.metrics import (
    LLMJudge,
    MetricResult,
    answer_correctness,
    answer_relevancy,
    citation_grounding_rate,
    citation_locator_accuracy,
    citation_offset_accuracy,
    context_precision,
    context_recall,
    context_relevance,
    faithfulness,
    hit_rate_at_k,
    is_abstention,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.runner import (
    DEFAULT_THRESHOLDS,
    LEXICAL_THRESHOLDS,
    EvalPrediction,
    EvaluationReport,
    PipelineAnswerProvider,
    RAGEvaluator,
    SampleResult,
    evaluate_sync,
    load_pages,
    save_report,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "LEXICAL_THRESHOLDS",
    "EvalPrediction",
    "EvaluationReport",
    "EvaluationSample",
    "LLMJudge",
    "MetricResult",
    "PipelineAnswerProvider",
    "RAGEvaluator",
    "SampleResult",
    "answer_correctness",
    "answer_relevancy",
    "citation_grounding_rate",
    "citation_locator_accuracy",
    "citation_offset_accuracy",
    "context_precision",
    "context_recall",
    "context_relevance",
    "evaluate_sync",
    "faithfulness",
    "hit_rate_at_k",
    "is_abstention",
    "load_dataset",
    "load_pages",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "relevant_chunk_ids",
    "save_report",
]
