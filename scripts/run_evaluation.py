#!/usr/bin/env python
"""Run the RAG quality evaluation suite over the sample contracts.

Ingests the committed sample documents, replays the CUAD-style ground-truth
dataset through the pipeline, scores every metric, and writes a JSON plus a
Markdown report. Exits non-zero when a metric falls below its threshold, so it
can be used as a CI gate.

    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --embedder minilm --out reports/eval
    python scripts/run_evaluation.py --judge --model gemma3n:e4b

The ``--judge`` flag switches the RAGAS decomposition metrics (faithfulness,
context recall, answer correctness, context relevance, answer relevancy) from
the deterministic lexical approximation to a local LLM judge. It is slower and
touches only the local Ollama endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings  # noqa: E402
from app.embeddings.deterministic import OfflineHashingEmbedder  # noqa: E402
from app.evaluation.dataset import EvaluationSample, load_dataset  # noqa: E402
from app.evaluation.metrics import LLMJudge  # noqa: E402
from app.evaluation.runner import (  # noqa: E402
    PipelineAnswerProvider,
    RAGEvaluator,
    save_report,
)
from app.llm.inference import LLMClient  # noqa: E402
from app.parsing.extractor import extract_text_from_pdf, extract_text_from_txt  # noqa: E402
from app.pipeline.triage import TriagePipeline  # noqa: E402
from app.vectorstore.store import VectorStore  # noqa: E402

SAMPLE_DIR = REPO_ROOT / "tests" / "sample_documents"
DEFAULT_DATASET = SAMPLE_DIR / "ground_truth.jsonl"

#: Sample documents ingested for evaluation, keyed by the slug used in the
#: dataset's ``document`` field.
DOCUMENTS: dict[str, str] = {
    "mutual_nda": "mutual_nda.txt",
    "service_agreement": "service_agreement.txt",
    "sparse_agreement": "sparse_agreement.txt",
}

#: slug -> file path actually ingested. Defaults to the committed samples; replaced
#: by ``--docs-map`` (written by scripts/prepare_cuad.py) for a CUAD evaluation.
DOC_PATHS: dict[str, Path] = {slug: SAMPLE_DIR / name for slug, name in DOCUMENTS.items()}

logger = logging.getLogger("contracter.evaluation")


async def build_pipeline(
    workdir: Path,
    embedder: Any,
    llm_client: LLMClient | None = None,
    model: str | None = None,
) -> TriagePipeline:
    """Create a pipeline and ingest every sample document.

    Args:
        workdir: Scratch directory for uploads and the vector index.
        embedder: Embedding engine to use.
        llm_client: Optional pre-built LLM client.
        model: Ollama model name when building a default client.

    Returns:
        The ingestion-ready pipeline.
    """
    config = Settings(
        UPLOAD_DIR=workdir / "uploads",
        CHROMA_DIR=workdir / "chroma",
        OLLAMA_MODEL=model or "gemma3n:e4b",
    )
    store = VectorStore(persist_dir=config.CHROMA_DIR, embedding_engine=embedder)
    pipeline = TriagePipeline(
        config=config,
        embedding_engine=embedder,
        vector_store=store,
        llm_client=llm_client or LLMClient(base_url=config.OLLAMA_BASE_URL, model_name=config.OLLAMA_MODEL),
    )

    for document_id, source in DOC_PATHS.items():
        filename = source.name
        upload_dir = Path(config.UPLOAD_DIR) / document_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / filename
        target.write_bytes(source.read_bytes())
        response = await pipeline.ingest_document(file_path=target, document_id=document_id)
        logger.info("ingested %s: %d chunks", document_id, response.num_chunks)

    return pipeline


def load_pages_safe(path: Path) -> list[dict[str, Any]]:
    """Extract pages from a sample document regardless of format.

    Args:
        path: Document path.

    Returns:
        Page dictionaries.
    """
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)
    return extract_text_from_txt(path)


async def evaluate(
    *,
    dataset_path: Path,
    embedder_kind: str,
    use_judge: bool,
    model: str,
    concurrency: int,
) -> tuple[Any, list[EvaluationSample]]:
    """Run the evaluation and return the report.

    Args:
        dataset_path: Ground-truth dataset path.
        embedder_kind: ``hash`` or ``minilm``.
        use_judge: Whether to use the local LLM judge.
        model: Ollama model name.
        concurrency: Number of samples evaluated in parallel.

    Returns:
        Tuple of (EvaluationReport, samples).
    """
    import tempfile

    if embedder_kind == "minilm":
        from app.embeddings.engine import EmbeddingEngine

        embedder: Any = EmbeddingEngine()
    else:
        embedder = OfflineHashingEmbedder()

    samples = load_dataset(dataset_path)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        pipeline = await build_pipeline(workdir, embedder, model=model)

        judge = LLMJudge(pipeline.llm_client) if use_judge else None
        evaluator = RAGEvaluator(judge=judge, embedder=embedder if use_judge else None)

        chunks_by_document = {doc: pipeline.vector_store.fetch_all(doc) for doc in DOC_PATHS}
        pages_by_document = {doc: load_pages_safe(path) for doc, path in DOC_PATHS.items()}

        report = await evaluator.evaluate(
            samples,
            PipelineAnswerProvider(pipeline),
            chunks_by_document=chunks_by_document,
            pages_by_document=pages_by_document,
            concurrency=concurrency,
        )

    return report, samples


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list.

    Returns:
        Process exit code: 0 on pass, 1 on threshold failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--embedder", choices=["hash", "minilm"], default="hash")
    parser.add_argument("--judge", action="store_true", help="Use the local LLM judge")
    parser.add_argument("--model", default="gemma3n:e4b", help="Ollama model for generation")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--docs-map",
        type=Path,
        default=None,
        help="JSON {slug: path} of documents to ingest (from scripts/prepare_cuad.py)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write reports to this .json path (a sibling .md is written too)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.docs_map is not None:
        import json as _json

        DOC_PATHS.clear()
        DOC_PATHS.update({k: Path(v) for k, v in _json.loads(args.docs_map.read_text()).items()})

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    report, samples = asyncio.run(
        evaluate(
            dataset_path=args.dataset,
            embedder_kind=args.embedder,
            use_judge=args.judge,
            model=args.model,
            concurrency=args.concurrency,
        )
    )

    print(report.to_markdown())
    print(
        f"\n{report.num_samples} samples, {len(report.aggregates)} metrics, "
        f"mode={'llm-judge' if args.judge else 'lexical'}, embedder={args.embedder}"
    )

    if args.out:
        written = save_report(report, args.out)
        print(f"reports written to {written} and {written.with_suffix('.md')}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
