"""Shared pytest fixtures for the Contracter test suite.

The suite is hermetic by default: no network, no model downloads, no Ollama
required. Tests that genuinely need the local model server or the real
sentence-transformers weights are marked and skip automatically when the
resource is unavailable.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from app.config import Settings
from app.embeddings.deterministic import DEFAULT_DIMENSION, OfflineHashingEmbedder
from app.evaluation.dataset import EvaluationSample, load_dataset
from app.pipeline.triage import TriagePipeline
from tests.support import (
    SAMPLE_DIR,
    GroundedAnalysisLLM,
    HallucinatingAnalysisLLM,
    ingest_sample_document,
    make_pipeline,
    make_settings,
)

# The committed sample documents and the stable ingestion slugs used by the
# evaluation dataset.
SAMPLE_DOCUMENTS: dict[str, str] = {
    "mutual_nda": "mutual_nda.txt",
    "service_agreement": "service_agreement.txt",
    "sparse_agreement": "sparse_agreement.txt",
}

GROUND_TRUTH_PATH = SAMPLE_DIR / "ground_truth.jsonl"


# ---------------------------------------------------------------------------
# Directory and dataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    """Path to the committed sample contract directory."""
    return SAMPLE_DIR


@pytest.fixture(scope="session")
def ground_truth_path() -> Path:
    """Path to the CUAD-style evaluation dataset."""
    return GROUND_TRUTH_PATH


@pytest.fixture(scope="session")
def dataset(ground_truth_path: Path) -> list[EvaluationSample]:
    """The parsed evaluation dataset."""
    return load_dataset(ground_truth_path)


@pytest.fixture(scope="session")
def pdf_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the sample PDFs once per session and return their directory."""
    from tests.sample_documents.generate_test_pdfs import generate_sample_pdfs

    out_dir = tmp_path_factory.mktemp("sample_pdfs")
    generate_sample_pdfs(out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Embedding fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def embedder() -> OfflineHashingEmbedder:
    """The deterministic offline embedder used by the hermetic tests."""
    return OfflineHashingEmbedder(dimension=DEFAULT_DIMENSION)


@pytest.fixture(scope="session")
def real_embedder() -> Any:
    """The production sentence-transformers embedder, or skip if unavailable.

    Skips rather than failing so the suite runs in a bare CI container, while
    still exercising the real embedding path on a developer machine.
    """
    try:
        from app.embeddings.engine import EmbeddingEngine
    except Exception as err:  # pragma: no cover - import-time dependency failures
        pytest.skip(f"sentence-transformers unavailable: {err}")

    try:
        return EmbeddingEngine()
    except Exception as err:  # pragma: no cover - model not cached / no GPU
        pytest.skip(f"embedding model not available locally: {err}")


# ---------------------------------------------------------------------------
# Pipeline fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_factory(
    tmp_path: Path,
    embedder: OfflineHashingEmbedder,
) -> Callable[..., TriagePipeline]:
    """Callable that builds an isolated pipeline rooted in ``tmp_path``."""

    def factory(
        llm_client: Any | None = None,
        embedding_engine: Any | None = None,
        **settings_overrides: Any,
    ) -> TriagePipeline:
        return make_pipeline(
            tmp_path,
            llm_client=llm_client,
            embedder=embedding_engine or embedder,
            **settings_overrides,
        )

    return factory


@pytest.fixture
def pipeline(pipeline_factory: Callable[..., TriagePipeline]) -> TriagePipeline:
    """A hermetic pipeline backed by the well-behaved grounded LLM double."""
    return pipeline_factory(llm_client=GroundedAnalysisLLM())


@pytest.fixture
def hallucinating_pipeline(
    pipeline_factory: Callable[..., TriagePipeline],
) -> TriagePipeline:
    """A hermetic pipeline backed by the deliberately unfaithful LLM double."""
    return pipeline_factory(llm_client=HallucinatingAnalysisLLM())


@pytest.fixture
def ingested_pipeline(pipeline: TriagePipeline) -> tuple[TriagePipeline, dict[str, str]]:
    """A pipeline with all three sample contracts already ingested.

    Returns:
        Tuple of (pipeline, mapping of ingestion slug to source filename).
    """
    for document_id, filename in SAMPLE_DOCUMENTS.items():
        ingest_sample_document(pipeline, document_id, filename)
    return pipeline, dict(SAMPLE_DOCUMENTS)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated application settings rooted in ``tmp_path``."""
    return make_settings(tmp_path)


# ---------------------------------------------------------------------------
# Network privacy fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def network_attempts() -> list[Any]:
    """List that the network blocker populates with blocked addresses."""
    return []


@pytest.fixture
def offline(network_attempts: list[Any]) -> Iterator[list[Any]]:
    """Block all outbound network access for the duration of a test.

    Loopback stays enabled so a locally hosted model server remains reachable;
    every external hostname and address is refused at the socket layer.
    """
    from tests.support import block_network

    with block_network(allow_loopback=True, attempts=network_attempts) as recorded:
        yield recorded


# ---------------------------------------------------------------------------
# Capability probes for optional tests
# ---------------------------------------------------------------------------


def _ollama_reachable(base_url: str = "http://127.0.0.1:11434", timeout: float = 0.5) -> bool:
    """Check whether a local Ollama server is listening.

    Args:
        base_url: Ollama base URL.
        timeout: Connect timeout in seconds.

    Returns:
        True when the port accepts a TCP connection.
    """
    host, _, port_text = base_url.split("//", 1)[-1].partition(":")
    port = int(port_text or "11434")
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    """Whether a local Ollama server is running."""
    return _ollama_reachable()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip marker-gated tests when their backing resource is missing.

    Args:
        config: Pytest configuration.
        items: Collected test items.
    """
    if not _ollama_reachable():
        skip_ollama = pytest.mark.skip(reason="local Ollama server is not reachable")
        for item in items:
            if "ollama" in item.keywords:
                item.add_marker(skip_ollama)
