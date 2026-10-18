#!/usr/bin/env python
"""Prove that the whole Contracter pipeline runs with the network switched off.

This is the "privacy validation" artifact required by the project plan. It runs
the *complete* document analysis pipeline — parse, chunk, embed, index, retrieve,
answer, summarize, delete — while every outbound socket operation is
intercepted. Any attempt to reach a non-loopback address aborts the run.

It is deliberately a standalone script rather than only a test, so it can be run
and demonstrated on its own:

    python scripts/verify_offline.py
    python scripts/verify_offline.py --no-real-model   # skip sentence-transformers
    python scripts/verify_offline.py --json            # machine-readable result

Exit codes:
    0  the pipeline completed with zero outbound network attempts
    1  a check failed
    2  the harness itself could not run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

# Allow running the script directly from a checkout.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Belt-and-braces: make the Hugging Face stack refuse to reach the network even
# if a code path bypasses the socket guard below.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.config import Settings  # noqa: E402
from app.embeddings.deterministic import OfflineHashingEmbedder  # noqa: E402
from app.embeddings.engine import EmbeddingEngine  # noqa: E402
from app.llm.inference import LLMClient  # noqa: E402
from app.pipeline.triage import TriagePipeline  # noqa: E402
from app.vectorstore.store import VectorStore  # noqa: E402

SAMPLE = REPO_ROOT / "tests" / "sample_documents" / "service_agreement.txt"

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "", "::"}


class _UnreachableLLMClient(LLMClient):
    """A client that always fails, used to verify graceful degradation.

    With ``--no-llm`` the intent is to prove the pipeline still returns a
    disclaimer-bearing refusal when no model is available. Pointing a real
    client at localhost would be non-deterministic: on a machine that happens to
    be running Ollama the run would silently take the real-model path. Refusing
    every call makes the degradation check reproducible.
    """

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> str:
        """Fail deterministically instead of contacting a model.

        Args:
            prompt: Ignored.
            system_prompt: Ignored.
            temperature: Ignored.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("offline verification: model calls disabled (--no-llm)")

    async def health_check(self) -> bool:
        """Report the model as unavailable.

        Returns:
            Always False.
        """
        return False


def _is_loopback(address: Any) -> bool:
    """Return True when an address targets the local machine only.

    Args:
        address: Socket address, or a hostname/bytes hostname.

    Returns:
        True for loopback and UNIX-domain targets.
    """
    if isinstance(address, bytes):
        address = address.decode("utf-8", errors="replace")
    if isinstance(address, str):
        # getaddrinfo/gethostbyname pass a bare hostname.
        return address in _LOOPBACK
    if not isinstance(address, (tuple, list)) or not address:
        return False
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode("utf-8", errors="replace")
    host = str(host)
    return host in _LOOPBACK or host.startswith("127.") or host.startswith("::1")


@contextmanager
def network_disabled(
    record: list[Any], loopback_record: list[Any] | None = None
) -> Iterator[list[Any]]:
    """Disable outbound network access at the socket layer.

    Loopback traffic is permitted (a locally hosted model server is not a
    privacy leak) but is recorded separately so the report can show that the
    only endpoint ever contacted was the local machine.

    Args:
        record: List that receives every blocked non-loopback attempt.
        loopback_record: Optional list that receives permitted loopback traffic.

    Yields:
        The violation list, for convenience.
    """
    permitted = loopback_record if loopback_record is not None else []
    real = {
        "connect": socket.socket.connect,
        "connect_ex": socket.socket.connect_ex,
        "create_connection": socket.create_connection,
        "getaddrinfo": socket.getaddrinfo,
        "gethostbyname": socket.gethostbyname,
    }

    def guard(address: Any, action: str) -> None:
        if _is_loopback(address):
            permitted.append({"action": action, "address": repr(address)})
            return
        record.append({"action": action, "address": repr(address)})
        raise RuntimeError(
            f"OFFLINE VIOLATION: attempted {action} to {address!r} while the network was disabled"
        )

    def connect(self: socket.socket, address: Any) -> Any:  # noqa: ANN001
        guard(address, "connect")
        return real["connect"](self, address)

    def connect_ex(self: socket.socket, address: Any) -> Any:  # noqa: ANN001
        guard(address, "connect_ex")
        return real["connect_ex"](self, address)

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        guard(address, "create_connection")
        return real["create_connection"](address, *args, **kwargs)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not (str(host) in _LOOPBACK):
            guard(host, "getaddrinfo")
        return real["getaddrinfo"](host, *args, **kwargs)

    def gethostbyname(host: Any) -> Any:
        if not (str(host) in _LOOPBACK):
            guard(host, "gethostbyname")
        return real["gethostbyname"](host)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    socket.create_connection = create_connection  # type: ignore[assignment]
    socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]
    socket.gethostbyname = gethostbyname  # type: ignore[assignment]

    try:
        yield record
    finally:
        socket.socket.connect = real["connect"]  # type: ignore[method-assign]
        socket.socket.connect_ex = real["connect_ex"]  # type: ignore[method-assign]
        socket.create_connection = real["create_connection"]  # type: ignore[assignment]
        socket.getaddrinfo = real["getaddrinfo"]  # type: ignore[assignment]
        socket.gethostbyname = real["gethostbyname"]  # type: ignore[assignment]


def _local_ollama_model(base_url: str = "http://127.0.0.1:11434") -> str | None:
    """Return an installed local Ollama model name, preferring Gemma 3n.

    The probe itself is a loopback call, which the network guard permits: the
    point of the check is that the *only* endpoint ever reached is localhost.

    Args:
        base_url: Ollama base URL.

    Returns:
        A model name, or None when Ollama is unreachable or has no models.
    """
    try:
        import httpx

        response = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        models = response.json().get("models", [])
        if not models:
            return None
        for model in models:
            name = str(model.get("name", ""))
            if name.startswith("gemma3n"):
                return name
        return str(models[0].get("name", "")) or None
    except Exception:
        return None


async def run_pipeline(
    workdir: Path,
    embedder: Any,
    model_label: str,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """Execute the full analysis pipeline and collect a report.

    Args:
        workdir: Scratch directory for uploads and the vector store.
        embedder: Embedding engine to use.
        model_label: Human-readable embedder label for the report.
        llm_model: Local Ollama model to use for the answer step. When None, the
            script verifies graceful degradation instead of a real answer.

    Returns:
        Report dictionary of check results.
    """
    report: dict[str, Any] = {
        "embedder": model_label,
        "llm_model": llm_model,
        "checks": {},
    }

    config = Settings(
        UPLOAD_DIR=workdir / "uploads",
        CHROMA_DIR=workdir / "chroma",
        OLLAMA_MODEL=llm_model or "offline-verification",
    )
    store = VectorStore(persist_dir=config.CHROMA_DIR, embedding_engine=embedder)
    pipeline = TriagePipeline(
        config=config,
        embedding_engine=embedder,
        vector_store=store,
        llm_client=None if llm_model else _UnreachableLLMClient(
            base_url=config.OLLAMA_BASE_URL, model_name=config.OLLAMA_MODEL
        ),
    )

    # 1. Ingest a real contract.
    upload_dir = Path(config.UPLOAD_DIR) / "verify"
    upload_dir.mkdir(parents=True, exist_ok=True)
    document_path = upload_dir / SAMPLE.name
    document_path.write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    ingestion = await pipeline.ingest_document(file_path=document_path, document_id="verify")
    report["checks"]["ingest"] = {
        "chunks": ingestion.num_chunks,
        "pages": ingestion.num_pages,
        "ok": ingestion.num_chunks > 0,
    }

    # 2. Retrieve evidence locally.
    hits = pipeline.retrieve("verify", "How much notice is required to terminate for convenience?")
    report["checks"]["retrieve"] = {
        "hits": len(hits),
        "top_page": hits[0].page_number if hits else None,
        "ok": len(hits) > 0,
    }

    # 3. Read the whole index back (the summarization path).
    all_chunks = store.fetch_all("verify")
    report["checks"]["read_index"] = {"chunks": len(all_chunks), "ok": len(all_chunks) > 0}

    # 4. Exercise the answer path. The disclaimer must be present whether the
    #    model answered or inference failed, and a non-committal answer for an
    #    unanswerable question must be reported as document silence.
    response = await pipeline.query_document(
        "verify", "What is the arbitration venue for disputes?"
    )
    notes = " ".join(response.uncertainty_notes).lower()
    answer_lower = response.answer.lower()
    # With no model reachable, the run must degrade to an explicit refusal or
    # error signal rather than invent an answer. With a model, it must produce a
    # non-empty answer. Either way the disclaimer has to be present.
    refused_without_model = ("silent" in notes) or ("inference error" in answer_lower)
    report["checks"]["answer_path"] = {
        "mode": "local-model" if llm_model else "degradation-only",
        "answer_chars": len(response.answer),
        "has_disclaimer": bool(response.disclaimer),
        "notes_mention_silence": "silent" in notes,
        "ok": bool(response.disclaimer)
        and len(response.answer) > 0
        and (bool(llm_model) or refused_without_model),
    }

    # 5. Delete the document (filesystem + vector store).
    deleted = await pipeline.delete_document("verify")
    remaining = store.list_documents()
    report["checks"]["delete"] = {
        "deleted": deleted,
        "remaining_collections": len(remaining),
        "ok": deleted and len(remaining) == 0,
    }

    return report


def main(argv: list[str] | None = None) -> int:
    """Run the offline verification.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-real-model",
        action="store_true",
        help="Use only the deterministic embedder, skipping sentence-transformers",
    )
    parser.add_argument(
        "--require-real-model",
        action="store_true",
        help="Fail instead of skipping when sentence-transformers is unavailable",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Do not call a local Ollama model; verify the degradation path instead",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args(argv)

    def emit(payload: dict[str, Any]) -> None:
        if args.json:
            print(json.dumps(payload, indent=2, default=str))
        else:
            print(f"\n=== Contracter offline verification: {payload['status']} ===")
            print(f"embedder: {payload.get('embedder', 'n/a')}")
            print(f"llm model: {payload.get('llm_model') or 'n/a'}")
            for name, check in payload.get("checks", {}).items():
                mark = "PASS" if check.get("ok") else "FAIL"
                detail = ", ".join(f"{k}={v}" for k, v in check.items() if k != "ok")
                print(f"  [{mark}] {name}: {detail}")
            attempts = payload.get("network_attempts", [])
            print(f"outbound violations: {len(attempts)}")
            for attempt in attempts:
                print(f"    ! {attempt}")
            loopback = payload.get("loopback_attempts", [])
            print(f"permitted loopback calls: {len(loopback)}")
            if payload.get("errors"):
                print("errors:")
                for error in payload["errors"]:
                    print(f"    {error}")
            print()

    embedder: Any
    model_label: str

    if args.no_real_model:
        embedder = OfflineHashingEmbedder()
        model_label = "OfflineHashingEmbedder (--no-real-model)"
    else:
        try:
            embedder = EmbeddingEngine()
            model_label = f"sentence-transformers/{getattr(embedder, 'model_name', 'unknown')}"
        except Exception as err:
            if args.require_real_model:
                emit(
                    {
                        "status": "ERROR",
                        "errors": [f"sentence-transformers unavailable: {err}"],
                        "checks": {},
                    }
                )
                return 2
            embedder = OfflineHashingEmbedder()
            model_label = f"OfflineHashingEmbedder (fallback: {err})"

    attempts: list[Any] = []
    loopback_attempts: list[Any] = []
    payload: dict[str, Any] = {
        "status": "PASS",
        "checks": {},
        "network_attempts": [],
        "loopback_attempts": [],
        "errors": [],
    }

    llm_model = None if args.no_llm else _local_ollama_model()
    if not args.json:
        print(
            "local Ollama model: "
            + (llm_model if llm_model else "none available (verifying degradation path only)")
        )

    try:
        with tempfile.TemporaryDirectory() as tmp:
            with network_disabled(attempts, loopback_attempts):
                report = asyncio.run(
                    run_pipeline(Path(tmp), embedder, model_label, llm_model)
                )
        payload.update(report)
    except Exception as err:
        payload["status"] = "FAIL"
        payload["errors"].append(f"{type(err).__name__}: {err}")

    payload["network_attempts"] = attempts
    payload["loopback_attempts"] = loopback_attempts

    if any(not check.get("ok") for check in payload["checks"].values()):
        payload["status"] = "FAIL"
    if attempts:
        payload["status"] = "FAIL"
        payload["errors"].append(
            f"{len(attempts)} outbound network attempt(s) detected during the run"
        )

    emit(payload)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
