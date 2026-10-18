"""End-to-end RAG evaluation harness for the Contracter legal triage pipeline.

The harness separates the two halves of the pipeline on purpose:

* **Retrieval metrics** (hit rate, recall@k, precision@k, MRR, nDCG@k) are scored
  against ground-truth chunk identifiers. A low score here means the vector
  store or the chunker is at fault.
* **Generation metrics** (faithfulness, answer relevancy, answer correctness,
  citation grounding) are scored against the produced answer. A low score here
  with healthy retrieval means the LLM is at fault.

That split is what makes a failing evaluation actionable instead of just a
number. Every metric is computed locally; the optional LLM judge talks only to
the local Ollama endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Protocol, Sequence

from pydantic import BaseModel, Field

from app.evaluation.dataset import EvaluationSample, relevant_chunk_ids
from app.evaluation.metrics import (
    LLMJudge,
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
from app.evaluation.text import mean

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default acceptance thresholds
# ---------------------------------------------------------------------------

#: Minimum acceptable aggregate score per metric. Calibrated to the guidance in
#: the RAG evaluation literature: faithfulness for a regulated domain such as
#: law is held to 0.9, and answer relevancy to roughly 0.75. These are the
#: targets for the **judged** (local LLM) mode.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.90,
    "answer_relevancy": 0.70,
    "context_relevance": 0.60,
    "context_precision": 0.60,
    "context_recall": 0.70,
    "answer_correctness": 0.50,
    "citation_grounding_rate": 0.95,
    "retrieval_hit_rate": 0.80,
    "retrieval_recall": 0.70,
    "retrieval_precision": 0.30,
    "retrieval_mrr": 0.60,
    "abstention_recall": 0.90,
    "over_abstention_rate": 0.20,
    "answer_clause_type_accuracy": 0.60,
}

#: Thresholds for the deterministic lexical mode. Only metrics that the lexical
#: judge measures reliably appear here.
#:
#: Three metrics are deliberately absent. ``answer_relevancy`` and
#: ``answer_correctness`` cannot recognize paraphrase or synonyms, so a correct
#: answer that does not reuse the contract's nouns scores far below what a
#: judge would award; they are gated in judged mode instead, and their maths is
#: verified directly by unit tests. ``abstention_recall``/``over_abstention_rate``
#: are excluded because detecting unanswerability is a modelling capability, not
#: a lexical one — they are asserted in dedicated tests against purpose-built
#: model doubles. All of them are still computed and reported.
LEXICAL_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.80,
    "context_relevance": 0.45,
    "context_precision": 0.60,
    "context_recall": 0.60,
    "citation_grounding_rate": 0.95,
    "citation_locator_accuracy": 0.95,
    "citation_offset_accuracy": 0.90,
    "retrieval_hit_rate": 0.85,
    "retrieval_recall": 0.80,
    # Calibrated against the deterministic lexical embedder, which ranks the
    # gold chunk first about half the time. The production
    # sentence-transformers embedder measured 0.90 on the same suite, so the
    # integration tests hold it to a stricter bar.
    "retrieval_precision": 0.45,
    "retrieval_mrr": 0.50,
    "retrieval_ndcg": 0.55,
}

#: Metrics where a *lower* value is better.
LOWER_IS_BETTER: frozenset[str] = frozenset({"over_abstention_rate", "hallucination_rate"})

#: Cutoff used for every @k retrieval metric.
DEFAULT_K: int = 5


# ---------------------------------------------------------------------------
# Prediction container
# ---------------------------------------------------------------------------


class EvalPrediction(BaseModel):
    """Everything the harness needs to score one answer.

    Attributes:
        answer: The synthesized answer text.
        contexts: The exact excerpts fed to the LLM, best match first.
        retrieved_chunk_ids: Identifiers of ``contexts``, used for retrieval metrics.
        clause_types: Clause categories returned by the pipeline.
        uncertainty_notes: Silence/ambiguity notes returned by the pipeline.
        cited_passages: Verbatim quotations the pipeline attached to its clauses.
        cited_passage_meta: Citations with page/offset metadata, for locator metrics.
    """

    answer: str = ""
    contexts: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    clause_types: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    cited_passages: list[str] = Field(default_factory=list)
    cited_passage_meta: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class AnswerProvider(Protocol):
    """Async callable that answers one evaluation sample."""

    def __call__(self, sample: EvaluationSample) -> Awaitable[EvalPrediction]:
        """Produce a prediction for a sample."""
        ...


# ---------------------------------------------------------------------------
# Pipeline-backed provider
# ---------------------------------------------------------------------------


class PipelineAnswerProvider:
    """Answers evaluation samples using a real :class:`TriagePipeline`.

    Retrieval and generation are obtained from a single pipeline call, so the
    scored answer was produced from exactly the scored evidence.
    """

    def __init__(self, pipeline: Any) -> None:
        """Initialize the provider.

        Args:
            pipeline: A ``TriagePipeline`` instance with an ingested document.
        """
        self.pipeline = pipeline

    async def __call__(self, sample: EvaluationSample) -> EvalPrediction:
        """Answer one sample and capture the evidence behind it.

        Args:
            sample: The evaluation sample; its ``document`` field names the
                document identifier to query.

        Returns:
            EvalPrediction capturing answer, contexts, clause types and citations.
        """
        document_id = sample.document
        try:
            response, chunks = await self.pipeline.query_document_with_evidence(
                document_id=document_id, question=sample.question
            )
        except Exception as err:  # pragma: no cover - transport failures
            logger.error("Pipeline failed for sample '%s': %s", sample.id, err)
            return EvalPrediction(error=str(err))

        cited_passages: list[str] = []
        cited_meta: list[dict[str, Any]] = []
        clause_types: list[str] = []
        for clause in response.clauses:
            clause_types.append(clause.clause_type)
            for passage in clause.cited_passages:
                cited_passages.append(passage.text)
                cited_meta.append(
                    {
                        "text": passage.text,
                        "page_number": passage.page_number,
                        "start_char": passage.start_char,
                        "end_char": passage.end_char,
                    }
                )

        return EvalPrediction(
            answer=response.answer,
            contexts=[chunk.content for chunk in chunks],
            retrieved_chunk_ids=[chunk.chunk_id for chunk in chunks],
            clause_types=clause_types,
            uncertainty_notes=list(response.uncertainty_notes),
            cited_passages=cited_passages,
            cited_passage_meta=cited_meta,
        )


# ---------------------------------------------------------------------------
# Reporting containers
# ---------------------------------------------------------------------------


class SampleResult(BaseModel):
    """Per-sample metric scores and the evidence needed to explain them."""

    sample_id: str
    question: str
    category: str = "qa"
    answerable: bool = True
    metrics: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    answer: str = ""
    error: str | None = None


class EvaluationReport(BaseModel):
    """Aggregate evaluation report with a CI pass/fail verdict."""

    num_samples: int = 0
    aggregates: dict[str, float] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    passed: bool = True
    samples: list[SampleResult] = Field(default_factory=list)
    metric_modes: dict[str, str] = Field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to JSON.

        Args:
            indent: JSON indentation width.

        Returns:
            JSON string.
        """
        return self.model_dump_json(indent=indent)

    def to_markdown(self) -> str:
        """Render a human-readable Markdown summary of the report.

        Returns:
            Markdown document with an aggregate table and failure list.
        """
        verdict = "PASSED ✅" if self.passed else "FAILED ❌"
        lines = [
            "# Contracter RAG Evaluation Report",
            "",
            f"**Samples evaluated:** {self.num_samples}",
            "",
            f"**Verdict:** {verdict}",
            "",
            "| Metric | Score | Threshold | Mode | Status |",
            "| --- | ---: | ---: | --- | --- |",
        ]
        for name in sorted(self.aggregates):
            score = self.aggregates[name]
            threshold = self.thresholds.get(name)
            mode = self.metric_modes.get(name, "lexical")
            if threshold is None:
                status = "—"
                threshold_text = "—"
            else:
                if name in LOWER_IS_BETTER:
                    ok = score <= threshold
                else:
                    ok = score >= threshold
                status = "✅" if ok else "❌"
                threshold_text = f"{threshold:.2f}"
            lines.append(f"| `{name}` | {score:.3f} | {threshold_text} | {mode} | {status} |")

        if self.failures:
            lines += ["", "## Threshold failures", ""]
            lines += [f"- `{failure}`" for failure in self.failures]

        lines += [
            "",
            "> This evaluation scores a document-analysis tool. It is not legal advice.",
            "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class _SampleInputs:
    """Internal bundle of per-sample inputs handed to the metric functions."""

    sample: EvaluationSample
    prediction: EvalPrediction
    gold_chunk_ids: list[str] = field(default_factory=list)
    source_pages: list[dict[str, Any]] = field(default_factory=list)


class RAGEvaluator:
    """Computes the full RAG metric suite over one or more evaluation samples."""

    def __init__(
        self,
        judge: LLMJudge | None = None,
        embedder: Any | None = None,
        *,
        k: int = DEFAULT_K,
        thresholds: dict[str, float] | None = None,
        use_llm: bool | None = None,
    ) -> None:
        """Configure the evaluator.

        Args:
            judge: Optional local LLM judge. Enables the RAGAS decomposition
                path for faithfulness, context recall and answer correctness.
            embedder: Optional embedding engine for the RAGAS-style answer
                relevancy metric.
            k: Cutoff for retrieval@k metrics.
            thresholds: Overrides for :data:`DEFAULT_THRESHOLDS`.
            use_llm: Force the judged path on or off. ``None`` enables it
                whenever a judge is available.
        """
        self.judge = judge
        self.embedder = embedder
        self.k = k
        self.use_llm = (judge is not None) if use_llm is None else use_llm
        if self.use_llm and judge is None:
            logger.warning("use_llm requested without a judge; falling back to lexical metrics")
            self.use_llm = False
        base = DEFAULT_THRESHOLDS if self.use_llm else LEXICAL_THRESHOLDS
        self.thresholds = {**base, **(thresholds or {})}

    @property
    def active_judge(self) -> LLMJudge | None:
        """The judge actually used, honouring :attr:`use_llm`."""
        return self.judge if self.use_llm else None

    async def evaluate_sample(
        self,
        sample: EvaluationSample,
        prediction: EvalPrediction,
        *,
        gold_chunk_ids: Sequence[str] | None = None,
        source_pages: Sequence[dict[str, Any]] | None = None,
    ) -> SampleResult:
        """Score a single sample across every applicable metric.

        Args:
            sample: Ground-truth sample.
            prediction: Pipeline output for the sample.
            gold_chunk_ids: Ground-truth relevant chunk identifiers.
            source_pages: Extracted pages, required for citation locator metrics.

        Returns:
            SampleResult holding metric scores and diagnostic details.
        """
        result = SampleResult(
            sample_id=sample.id,
            question=sample.question,
            category=sample.category,
            answerable=sample.answerable,
            answer=prediction.answer[:2000],
            error=prediction.error,
        )

        if prediction.error:
            result.details["error"] = prediction.error
            return result

        contexts = prediction.contexts
        judge = self.active_judge

        # --- Retrieval metrics (need ground-truth chunk ids) -------------
        gold_ids = list(gold_chunk_ids or [])
        retrieved = list(prediction.retrieved_chunk_ids)
        if gold_ids and retrieved:
            result.metrics["retrieval_hit_rate"] = hit_rate_at_k(retrieved, gold_ids, self.k)
            result.metrics["retrieval_recall"] = recall_at_k(retrieved, gold_ids, self.k)
            # Precision is measured at R, the number of relevant chunks, because
            # with a single relevant passage per question precision@k is capped
            # at 1/k and says nothing.
            precision_cutoff = min(self.k, max(1, len(gold_ids)))
            result.metrics["retrieval_precision"] = precision_at_k(
                retrieved, gold_ids, precision_cutoff
            )
            result.metrics["retrieval_mrr"] = reciprocal_rank(retrieved, gold_ids)
            result.metrics["retrieval_ndcg"] = ndcg_at_k(retrieved, gold_ids, self.k)
            result.details["gold_chunk_ids"] = gold_ids
            result.details["retrieved_chunk_ids"] = retrieved
            result.details["precision_cutoff"] = precision_cutoff

        # --- Answerable cases: faithfulness & relevancy ------------------
        if sample.answerable:
            faith = await faithfulness(prediction.answer, contexts, judge)
            levity = await answer_relevancy(
                sample.question, prediction.answer, judge, self.embedder
            )
            correctness = await answer_correctness(prediction.answer, sample.reference_answer, judge)

            result.metrics["faithfulness"] = faith.score
            result.metrics["answer_relevancy"] = levity.score
            result.metrics["answer_correctness"] = correctness.score
            result.metrics["hallucination_rate"] = 1.0 - faith.score
            result.details["faithfulness"] = faith.details
            result.details["answer_relevancy"] = levity.details
            result.details["answer_correctness"] = correctness.details

            if contexts:
                crel = await context_relevance(sample.question, contexts, judge)
                cprec = await context_precision(sample.question, contexts, judge)
                result.metrics["context_relevance"] = crel.score
                result.metrics["context_precision"] = cprec.score
                result.details["context_relevance"] = crel.details
                result.details["context_precision"] = cprec.details

            if sample.reference_answer:
                crec = await context_recall(sample.reference_answer, contexts, judge)
                result.metrics["context_recall"] = crec.score
                result.details["context_recall"] = crec.details

            # Clause-type coverage: did the pipeline surface the expected category?
            if sample.relevant_clause_types:
                produced = {t.strip().lower() for t in prediction.clause_types}
                expected = {t.strip().lower() for t in sample.relevant_clause_types}
                result.metrics["answer_clause_type_accuracy"] = (
                    len(produced & expected) / len(expected)
                )
                result.details["expected_clause_types"] = sorted(expected)
                result.details["produced_clause_types"] = sorted(produced)

            # Calibrated uncertainty in reverse: answering an answerable question.
            result.metrics["over_abstention_rate"] = (
                1.0 if is_abstention(prediction.answer) else 0.0
            )

        # --- Unanswerable cases: calibrated abstention --------------------
        else:
            abstained = is_abstention(prediction.answer)
            notes_mention_silence = any(
                is_abstention(note) for note in prediction.uncertainty_notes
            )
            result.metrics["abstention_recall"] = 1.0 if abstained else 0.0
            result.metrics["abstention_calibrated"] = (
                1.0 if (abstained and notes_mention_silence) else 0.0
            )
            # An unanswerable case must not invent clauses.
            result.metrics["fabricated_clause_rate"] = (
                1.0 if prediction.clause_types else 0.0
            )
            result.details["abstained"] = abstained
            result.details["silence_noted"] = notes_mention_silence

        # --- Citation verification (always applicable) --------------------
        if prediction.cited_passages:
            grounding = citation_grounding_rate(prediction.cited_passages, contexts)
            result.metrics["citation_grounding_rate"] = grounding.score
            result.details["citation_grounding"] = grounding.details

            if source_pages:
                locator = citation_locator_accuracy(prediction.cited_passage_meta, source_pages)
                offsets = citation_offset_accuracy(prediction.cited_passage_meta, source_pages)
                result.metrics["citation_locator_accuracy"] = locator.score
                result.metrics["citation_offset_accuracy"] = offsets.score
                result.details["citation_locator"] = locator.details
                result.details["citation_offsets"] = offsets.details

        return result

    async def evaluate(
        self,
        samples: Sequence[EvaluationSample],
        provider: AnswerProvider,
        *,
        chunks_by_document: dict[str, Sequence[Any]] | None = None,
        pages_by_document: dict[str, Sequence[dict[str, Any]]] | None = None,
        concurrency: int = 1,
    ) -> EvaluationReport:
        """Evaluate a dataset and aggregate the metric scores.

        Args:
            samples: Ground-truth evaluation samples.
            provider: Async callable producing a prediction per sample.
            chunks_by_document: Full chunk inventory per document, used to
                resolve ground-truth chunk identifiers for retrieval metrics.
            pages_by_document: Extracted pages per document, used for the
                citation locator metrics.
            concurrency: Maximum number of samples evaluated in parallel.

        Returns:
            EvaluationReport with per-metric aggregates and a pass/fail verdict.
        """
        chunks_by_document = chunks_by_document or {}
        pages_by_document = pages_by_document or {}

        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def _run(sample: EvaluationSample) -> SampleResult:
            async with semaphore:
                prediction = await provider(sample)
                gold_ids: list[str] = []
                inventory = list(chunks_by_document.get(sample.document, []))
                if not inventory:
                    inventory = list(chunks_by_document.get("", []))
                if inventory:
                    gold_ids = relevant_chunk_ids(sample, inventory)
                pages = list(pages_by_document.get(sample.document, []))
                return await self.evaluate_sample(
                    sample,
                    prediction,
                    gold_chunk_ids=gold_ids,
                    source_pages=pages,
                )

        results: list[SampleResult] = []
        if concurrency <= 1:
            for sample in samples:
                results.append(await _run(sample))
        else:
            gathered = await asyncio.gather(*(_run(sample) for sample in samples))
            results = list(gathered)

        aggregates = self._aggregate(results)
        failures = self._check_thresholds(aggregates)
        modes = self._metric_modes()

        return EvaluationReport(
            num_samples=len(results),
            aggregates=aggregates,
            thresholds=self.thresholds,
            failures=failures,
            passed=not failures,
            samples=results,
            metric_modes=modes,
        )

    # -- internals ---------------------------------------------------------

    def _aggregate(self, results: Sequence[SampleResult]) -> dict[str, float]:
        """Average every metric across the samples that produced it.

        Args:
            results: Per-sample results.

        Returns:
            Mapping of metric name to mean score, including derived roll-ups.
        """
        buckets: dict[str, list[float]] = {}
        for result in results:
            for name, value in result.metrics.items():
                buckets.setdefault(name, []).append(float(value))

        aggregates = {name: round(mean(values), 6) for name, values in buckets.items()}

        # Hallucination rate is already recorded per sample; keep the mean
        # (not the complement of the mean) so its own threshold stays meaningful.
        if "hallucination_rate" in buckets:
            aggregates["hallucination_rate"] = round(mean(buckets["hallucination_rate"]), 6)

        return aggregates

    def _check_thresholds(self, aggregates: dict[str, float]) -> list[str]:
        """Compare aggregates against thresholds and describe every failure.

        Args:
            aggregates: Aggregated metric scores.

        Returns:
            List of human-readable failure descriptions; empty when all pass.
        """
        failures: list[str] = []
        for name, threshold in self.thresholds.items():
            if name not in aggregates:
                continue
            score = aggregates[name]
            if name in LOWER_IS_BETTER:
                if score > threshold:
                    failures.append(
                        f"{name}={score:.3f} exceeds the maximum allowed {threshold:.2f}"
                    )
            elif score < threshold:
                failures.append(
                    f"{name}={score:.3f} is below the required minimum {threshold:.2f}"
                )
        return failures

    def _metric_modes(self) -> dict[str, str]:
        """Report whether each metric ran in judged or lexical mode.

        Returns:
            Mapping of metric name to mode label.
        """
        judged_mode = "llm" if self.active_judge is not None else "lexical"
        deterministic = {
            "citation_grounding_rate",
            "citation_locator_accuracy",
            "citation_offset_accuracy",
            "retrieval_hit_rate",
            "retrieval_recall",
            "retrieval_precision",
            "retrieval_mrr",
            "retrieval_ndcg",
        }
        return {
            name: ("deterministic" if name in deterministic else judged_mode)
            for name in self.thresholds
        }


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def evaluate_sync(
    samples: Sequence[EvaluationSample],
    provider: AnswerProvider,
    *,
    chunks_by_document: dict[str, Sequence[Any]] | None = None,
    pages_by_document: dict[str, Sequence[dict[str, Any]]] | None = None,
    evaluator: RAGEvaluator | None = None,
) -> EvaluationReport:
    """Run :meth:`RAGEvaluator.evaluate` from synchronous code.

    Args:
        samples: Ground-truth evaluation samples.
        provider: Async prediction provider.
        chunks_by_document: Chunk inventory per document.
        pages_by_document: Extracted pages per document.
        evaluator: Pre-configured evaluator; a lexical one is built by default.

    Returns:
        The aggregated EvaluationReport.
    """
    active = evaluator or RAGEvaluator()
    return asyncio.run(
        active.evaluate(
            samples,
            provider,
            chunks_by_document=chunks_by_document,
            pages_by_document=pages_by_document,
        )
    )


def load_pages(document_path: str | Path) -> list[dict[str, Any]]:
    """Extract page-level text from a sample document for locator metrics.

    Args:
        document_path: Path to a PDF, TXT or Markdown document.

    Returns:
        Page dictionaries as produced by ``app.parsing.extractor``.
    """
    from app.parsing.extractor import extract_text_from_pdf, extract_text_from_txt

    path = Path(document_path)
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)
    return extract_text_from_txt(path)


def save_report(report: EvaluationReport, path: str | Path) -> Path:
    """Write a report to disk as JSON alongside a Markdown rendering.

    Args:
        report: The evaluation report.
        path: Destination ``.json`` path; a sibling ``.md`` file is written too.

    Returns:
        Path to the written JSON report.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(json.loads(report.to_json()), indent=2), encoding="utf-8")
    destination.with_suffix(".md").write_text(report.to_markdown(), encoding="utf-8")
    return destination
