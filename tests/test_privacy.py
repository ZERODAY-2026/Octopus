"""Zero-network privacy validation for the Contracter pipeline.

The project's first guarantee is that no document text, embedding or telemetry
leaves the machine. These tests do not ask the code to behave; they make the
network unusable and then run the whole pipeline through it. Anything that
silently reached out would raise here rather than pass unnoticed.

The socket guard is validated against itself first: a test that only ever sees
the happy path proves nothing if the guard is inert.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from app.embeddings.deterministic import OfflineHashingEmbedder
from app.models import QueryResponse
from app.pipeline.triage import TriagePipeline
from app.vectorstore.store import VectorStore
from tests.support import (
    NetworkBlockedError,
    block_network,
    ingest_sample_document_async,
    make_pipeline,
    run_async,
)

pytestmark = pytest.mark.privacy

PUBLIC_ADDRESS = ("93.184.216.34", 443)  # example.com, numeric so no DNS needed


# ---------------------------------------------------------------------------
# The guard itself
# ---------------------------------------------------------------------------


def test_guard_blocks_outbound_tcp(offline: list[object]) -> None:
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(PUBLIC_ADDRESS, timeout=0.1)
    assert offline, "the blocked attempt must be recorded"


def test_guard_blocks_external_dns(offline: list[object]) -> None:
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("example.com", 443)
    assert offline


def test_guard_permits_loopback() -> None:
    with block_network(allow_loopback=True) as recorded:
        socket.getaddrinfo("localhost", 80)
    assert recorded == []


# ---------------------------------------------------------------------------
# The pipeline runs end to end with the network down
# ---------------------------------------------------------------------------


def test_full_pipeline_runs_offline_without_outbound_attempts(
    tmp_path: Path,
    offline: list[object],
) -> None:
    pipeline = make_pipeline(tmp_path)  # grounded double, offline embedder

    response = run_async(
        _exercise_pipeline(pipeline, "ndaservice", "mutual_nda.txt")
    )

    assert isinstance(response, QueryResponse)
    assert offline == [], f"outbound attempts detected: {offline}"


async def _exercise_pipeline(
    pipeline: TriagePipeline, document_id: str, sample: str
) -> QueryResponse:
    """Drive ingest -> retrieve -> answer -> summarize -> read -> delete.

    Args:
        pipeline: Hermetic pipeline under test.
        document_id: Identifier to ingest under.
        sample: Sample document file name.

    Returns:
        The answer produced by the query step.
    """
    ingestion = await ingest_sample_document_async(pipeline, document_id, sample)
    assert ingestion.num_chunks > 0

    hits = pipeline.retrieve(document_id, "How much notice is required to terminate?")
    assert hits

    response = await pipeline.query_document(
        document_id, "How much notice is required to terminate for convenience?"
    )

    summary = await pipeline.summarize_document(document_id)
    assert summary.overall_summary

    inventory = pipeline.vector_store.fetch_all(document_id)
    assert inventory

    deleted = await pipeline.delete_document(document_id)
    assert deleted
    return response


def test_ingestion_and_retrieval_need_no_network(
    tmp_path: Path, offline: list[object]
) -> None:
    pipeline = make_pipeline(tmp_path)
    run_async(ingest_sample_document_async(pipeline, "nda", "mutual_nda.txt"))
    hits = pipeline.retrieve("nda", "confidential information obligations")
    assert hits
    assert offline == []


def test_offline_embedder_is_deterministic_and_offline(offline: list[object]) -> None:
    embedder = OfflineHashingEmbedder()
    first = embedder.encode(["termination for convenience upon thirty days"])
    second = embedder.encode(["termination for convenience upon thirty days"])

    assert first == second
    assert embedder.dimension == len(first[0])
    assert offline == []


def test_vector_store_disables_telemetry(tmp_path: Path, offline: list[object]) -> None:
    store = VectorStore(persist_dir=tmp_path / "chroma", embedding_engine=OfflineHashingEmbedder())
    # A local PersistentClient with anonymised telemetry switched off.
    assert store.client is not None
    settings = getattr(store.client, "_settings", None)
    if settings is not None:  # pragma: no cover - version dependent attribute
        assert getattr(settings, "anonymized_telemetry", False) is False
    assert offline == []


# ---------------------------------------------------------------------------
# The standalone privacy artifact
# ---------------------------------------------------------------------------


def test_verify_offline_script_reports_zero_violations(tmp_path: Path) -> None:
    # Run the shipped privacy-validation script's pipeline with the network
    # disabled and the degradation path forced, then assert nothing escaped.
    from scripts.verify_offline import network_disabled, run_pipeline

    violations: list[object] = []
    loopback: list[object] = []
    with network_disabled(violations, loopback):
        report = run_async(
            run_pipeline(tmp_path, OfflineHashingEmbedder(), "hash-test", llm_model=None)
        )

    assert violations == []
    assert all(check["ok"] for check in report["checks"].values()), report["checks"]
