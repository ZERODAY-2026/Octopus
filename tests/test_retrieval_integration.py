"""Integration tests over the committed CUAD-style sample contracts.

These tests run the real pipeline (chunker, vector store, retrieval, synthesis
with a grounded model double) against the ground-truth dataset, and check the
four links the evaluation harness scores separately:

* retrieval finds the chunk that justifies each expert answer;
* every stored chunk maps back to an exact slice of its source page;
* the pipeline surfaces the expected clause category for a question;
* every emitted citation is a genuine, correctly-located slice of the document.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.dataset import EvaluationSample, load_dataset, relevant_chunk_ids
from app.evaluation.runner import (
    LEXICAL_THRESHOLDS,
    PipelineAnswerProvider,
    RAGEvaluator,
    evaluate_sync,
    load_pages,
)
from app.evaluation.text import contains_verbatim
from app.models import QueryResponse
from app.pipeline.triage import TriagePipeline
from tests.conftest import SAMPLE_DOCUMENTS
from tests.support import (
    SAMPLE_DIR,
    GroundedAnalysisLLM,
    ingest_sample_document,
    ingest_sample_document_async,
    make_pipeline,
    run_async,
)

pytestmark = pytest.mark.slow


def _pages(document: str) -> list[dict[str, object]]:
    return load_pages(SAMPLE_DIR / SAMPLE_DOCUMENTS[document])


def _gold_ids(sample: EvaluationSample, pipeline: TriagePipeline) -> list[str]:
    inventory = pipeline.vector_store.fetch_all(sample.document)
    return relevant_chunk_ids(sample, inventory)


# ---------------------------------------------------------------------------
# Dataset shape
# ---------------------------------------------------------------------------


def test_dataset_loads_and_covers_every_sample_document(
    dataset: list[EvaluationSample],
) -> None:
    assert len(dataset) >= 20
    covered = {sample.document for sample in dataset}
    assert covered <= set(SAMPLE_DOCUMENTS)
    assert covered == set(SAMPLE_DOCUMENTS)
    # The suite must exercise both answering and calibrated abstention.
    assert any(sample.answerable for sample in dataset)
    assert any(not sample.answerable for sample in dataset)


def test_answerable_samples_carry_ground_truth(
    dataset: list[EvaluationSample],
) -> None:
    for sample in dataset:
        if sample.answerable:
            assert sample.reference_answer, sample.id
            assert sample.gold_passages or sample.relevant_keywords, sample.id


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------


def test_every_answerable_sample_resolves_to_a_gold_chunk(
    dataset: list[EvaluationSample], ingested_pipeline: tuple[TriagePipeline, dict[str, str]]
) -> None:
    pipeline, _ = ingested_pipeline
    for sample in dataset:
        if not sample.answerable:
            continue
        assert _gold_ids(sample, pipeline), f"no ground-truth chunk for {sample.id}"


def test_retrieval_scores_on_the_sample_suite(
    dataset: list[EvaluationSample], ingested_pipeline: tuple[TriagePipeline, dict[str, str]]
) -> None:
    from app.evaluation.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, reciprocal_rank

    pipeline, _ = ingested_pipeline
    hits: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []

    for sample in dataset:
        if not sample.answerable:
            continue
        gold = _gold_ids(sample, pipeline)
        assert gold, sample.id
        retrieved = [chunk.chunk_id for chunk in pipeline.retrieve(sample.document, sample.question)]
        # The hash embedder is deterministic, so these bounds are exact rather
        # than fuzzy. They are set below the measured values to leave headroom.
        hits.append(hit_rate_at_k(retrieved, gold, 5))
        recalls.append(recall_at_k(retrieved, gold, 5))
        mrrs.append(reciprocal_rank(retrieved, gold))
        ndcgs.append(ndcg_at_k(retrieved, gold, 5))

    assert sum(hits) / len(hits) == pytest.approx(1.0)
    assert sum(recalls) / len(recalls) >= 0.80
    assert sum(mrrs) / len(mrrs) >= 0.60
    assert sum(ndcgs) / len(ndcgs) >= 0.60


def test_stored_chunks_are_exact_slices_of_their_page(
    ingested_pipeline: tuple[TriagePipeline, dict[str, str]]
) -> None:
    pipeline, documents = ingested_pipeline
    for document in documents:
        pages = {int(page["page_number"]): str(page["text"]) for page in _pages(document)}
        inventory = pipeline.vector_store.fetch_all(document)
        assert inventory
        indices = [chunk.chunk_index for chunk in inventory]
        assert indices == sorted(indices)
        for chunk in inventory:
            assert chunk.chunk_id
            page_text = pages[chunk.page_number]
            assert page_text[chunk.start_char : chunk.end_char] == chunk.content


# ---------------------------------------------------------------------------
# Clause mapping
# ---------------------------------------------------------------------------


def test_pipeline_surfaces_expected_clause_categories(
    dataset: list[EvaluationSample], ingested_pipeline: tuple[TriagePipeline, dict[str, str]]
) -> None:
    pipeline, _ = ingested_pipeline
    scored = 0
    matched = 0

    for sample in dataset:
        if not (sample.answerable and sample.relevant_clause_types):
            continue
        response = run_async(pipeline.query_document(sample.document, sample.question))
        produced = {clause.clause_type.lower() for clause in response.clauses}
        expected = {clause.lower() for clause in sample.relevant_clause_types}
        scored += 1
        if produced & expected:
            matched += 1

    assert scored
    # Measured ~0.78 on the sample suite; hold the bar below that.
    assert matched / scored >= 0.60


# ---------------------------------------------------------------------------
# The citation guarantee
# ---------------------------------------------------------------------------


def test_every_answerable_answer_emits_only_located_citations(
    dataset: list[EvaluationSample], ingested_pipeline: tuple[TriagePipeline, dict[str, str]]
) -> None:
    pipeline, _ = ingested_pipeline

    for sample in dataset:
        if not sample.answerable:
            continue
        response = run_async(pipeline.query_document(sample.document, sample.question))
        if not response.clauses:
            continue
        contexts = [chunk.content for chunk in pipeline.retrieve(sample.document, sample.question)]
        pages = {int(page["page_number"]): str(page["text"]) for page in _pages(sample.document)}

        for clause in response.clauses:
            assert clause.cited_passages, f"{sample.id}: cited clause with no passage"
            for passage in clause.cited_passages:
                # 1. The quotation exists in the retrieved evidence...
                assert any(
                    contains_verbatim(passage.text, context) for context in contexts
                ), f"{sample.id}: citation not in evidence: {passage.text!r}"
                # 2. ...and its offsets select exactly that text on the page.
                page_text = pages[passage.page_number]
                assert page_text[passage.start_char : passage.end_char] == passage.text


def test_unverifiable_quotations_are_withheld_not_shown(tmp_path: Path) -> None:
    # A model that invents a citation must not have that citation surface.
    from tests.support import HallucinatingAnalysisLLM

    pipeline = make_pipeline(tmp_path, llm_client=HallucinatingAnalysisLLM())
    ingest_sample_document(pipeline, "nda", "mutual_nda.txt")
    response: QueryResponse = run_async(
        pipeline.query_document("nda", "What penalty applies to a breach?")
    )

    contexts = [chunk.content for chunk in pipeline.retrieve("nda", "What penalty applies to a breach?")]
    for clause in response.clauses:
        for passage in clause.cited_passages:
            assert any(contains_verbatim(passage.text, context) for context in contexts)


# ---------------------------------------------------------------------------
# PDF end-to-end
# ---------------------------------------------------------------------------


def test_pdf_ingestion_and_query_round_trip(
    tmp_path: Path, pdf_dir: Path, embedder: object
) -> None:
    pipeline = make_pipeline(tmp_path, embedder=embedder)
    source = pdf_dir / "mutual_nda.pdf"
    assert source.is_file()

    target_dir = Path(pipeline.config.UPLOAD_DIR) / "nda-pdf"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    target.write_bytes(source.read_bytes())

    ingestion = run_async(pipeline.ingest_document(file_path=target, document_id="nda-pdf"))
    assert ingestion.num_chunks > 0
    assert ingestion.num_pages >= 1

    response = run_async(
        pipeline.query_document("nda-pdf", "How long do confidentiality obligations survive?")
    )
    assert response.answer
    assert response.disclaimer


# ---------------------------------------------------------------------------
# Whole-suite evaluation with the grounded double (CI gate)
# ---------------------------------------------------------------------------


def test_lexical_evaluation_passes_with_the_grounded_double(
    dataset: list[EvaluationSample], ingested_pipeline: tuple[TriagePipeline, dict[str, str]]
) -> None:
    pipeline, documents = ingested_pipeline
    evaluator = RAGEvaluator()  # lexical mode: no judge, no model, no network

    report = evaluate_sync(
        dataset,
        PipelineAnswerProvider(pipeline),
        chunks_by_document={doc: pipeline.vector_store.fetch_all(doc) for doc in documents},
        pages_by_document={doc: _pages(doc) for doc in documents},
        evaluator=evaluator,
    )

    assert report.passed, report.failures
    # The citation guarantee must be perfect, not merely above threshold.
    assert report.aggregates["citation_grounding_rate"] == pytest.approx(1.0)
    assert report.aggregates["citation_locator_accuracy"] == pytest.approx(1.0)
    assert report.aggregates["citation_offset_accuracy"] == pytest.approx(1.0)
    assert "faithfulness" in report.aggregates
    for metric, threshold in LEXICAL_THRESHOLDS.items():
        assert report.aggregates[metric] >= threshold, metric


@pytest.mark.embedding
def test_real_embedder_retrieval_is_strong(
    dataset: list[EvaluationSample], tmp_path: Path, real_embedder: object
) -> None:
    from app.evaluation.metrics import reciprocal_rank

    pipeline = make_pipeline(tmp_path, embedder=real_embedder)
    for document, filename in SAMPLE_DOCUMENTS.items():
        run_async(ingest_sample_document_async(pipeline, document, filename))

    reciprocal_ranks: list[float] = []
    for sample in dataset:
        if not sample.answerable:
            continue
        gold = _gold_ids(sample, pipeline)
        if not gold:
            continue
        retrieved = [chunk.chunk_id for chunk in pipeline.retrieve(sample.document, sample.question)]
        reciprocal_ranks.append(reciprocal_rank(retrieved, gold))

    assert sum(reciprocal_ranks) / len(reciprocal_ranks) >= 0.85
