"""FastAPI router providing endpoints for legal document upload, query, summary, and health checks."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
from typing import Any
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.config import settings
from app.models import (
    DocumentSummary,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.pipeline.triage import TriagePipeline
from app.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["legal-triage"])

_pipeline_instance: TriagePipeline | None = None


def get_pipeline() -> TriagePipeline:
    """FastAPI dependency provider for the TriagePipeline singleton."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = TriagePipeline()
    return _pipeline_instance


def set_pipeline(pipeline: TriagePipeline | None) -> None:
    """Set or reset the pipeline singleton instance (primarily for tests)."""
    global _pipeline_instance
    _pipeline_instance = pipeline


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    pipeline: TriagePipeline = Depends(get_pipeline),
) -> UploadResponse:
    """Accept and ingest a legal document (PDF or TXT).

    Validates file extension, verifies size under 50MB, securely stores the file
    under data/uploads/{document_id}/, extracts text and clauses, indexes into
    the vector database, and returns ingestion statistics.
    """
    raw_filename = file.filename or ""

    # Security: Detect path traversal attempts
    if ".." in raw_filename or "/" in raw_filename or "\\" in raw_filename:
        logger.warning("Path traversal attempt in uploaded filename: '%s'", raw_filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: path traversal sequences are forbidden.",
        )

    safe_filename = Path(raw_filename).name
    if not safe_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename cannot be empty.",
        )

    # Validate file extension
    ext = Path(safe_filename).suffix.lower()
    if ext not in (".pdf", ".txt"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Only .pdf and .txt files are allowed.",
        )

    # Read and validate file size (max 50MB)
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    chunk_size = 1024 * 1024  # 1MB
    total_size = 0
    file_bytes = bytearray()

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB.",
            )
        file_bytes.extend(chunk)

    document_id = str(uuid.uuid4())
    doc_dir = Path(settings.UPLOAD_DIR) / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    saved_path = doc_dir / safe_filename

    try:
        saved_path.write_bytes(file_bytes)
        upload_response = await pipeline.ingest_document(
            file_path=saved_path,
            document_id=document_id,
        )
        return upload_response
    except HTTPException:
        if doc_dir.exists():
            shutil.rmtree(doc_dir, ignore_errors=True)
        raise
    except Exception as err:
        logger.error("Failed to ingest uploaded document: %s", err)
        if doc_dir.exists():
            shutil.rmtree(doc_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process and ingest document: {err}",
        )


@router.post("/query", response_model=QueryResponse)
async def query_document(
    request: QueryRequest,
    pipeline: TriagePipeline = Depends(get_pipeline),
) -> QueryResponse:
    """Submit a query against an ingested legal document.

    Retrieves semantically relevant excerpts, synthesizes plain-language
    findings with verbatim citations, notes document silence/uncertainty,
    and appends a mandatory legal disclaimer.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query question must not be empty.",
        )

    doc_id = request.document_id.strip()
    sanitized = VectorStore._sanitize_collection_name(doc_id)
    known_docs = pipeline.list_documents()
    known_cols = pipeline.vector_store.list_documents()

    if doc_id not in known_docs and sanitized not in known_cols and doc_id not in known_cols:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    return await pipeline.query_document(document_id=doc_id, question=question)


@router.get("/documents", response_model=list[str])
async def list_documents(
    pipeline: TriagePipeline = Depends(get_pipeline),
) -> list[str]:
    """List all ingested document IDs available for query and summarization."""
    return pipeline.list_documents()


@router.get("/documents/{document_id}/summary", response_model=DocumentSummary)
async def get_document_summary(
    document_id: str,
    pipeline: TriagePipeline = Depends(get_pipeline),
) -> DocumentSummary:
    """Retrieve an automated legal summary of an ingested document."""
    doc_id = document_id.strip()
    sanitized = VectorStore._sanitize_collection_name(doc_id)
    known_docs = pipeline.list_documents()
    known_cols = pipeline.vector_store.list_documents()

    if doc_id not in known_docs and sanitized not in known_cols and doc_id not in known_cols:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    return await pipeline.summarize_document(document_id=doc_id)


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    pipeline: TriagePipeline = Depends(get_pipeline),
) -> dict[str, Any]:
    """Delete an ingested document, its vector index, and stored files."""
    doc_id = document_id.strip()
    sanitized = VectorStore._sanitize_collection_name(doc_id)
    known_docs = pipeline.list_documents()
    known_cols = pipeline.vector_store.list_documents()

    if doc_id not in known_docs and sanitized not in known_cols and doc_id not in known_cols:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    success = await pipeline.delete_document(document_id=doc_id)
    return {"success": success, "document_id": doc_id}


@router.get("/health")
async def health_check(
    pipeline: TriagePipeline = Depends(get_pipeline),
) -> dict[str, Any]:
    """Perform health checks on local Ollama LLM, ChromaDB vector storage, and system metrics."""
    ollama_ok = await pipeline.llm_client.health_check()
    chromadb_ok = False
    doc_count = 0
    try:
        collections = pipeline.vector_store.list_documents()
        doc_count = len(collections)
        chromadb_ok = True
    except Exception as err:
        logger.warning("ChromaDB health probe failed: %s", err)

    overall_status = "healthy" if (ollama_ok and chromadb_ok) else ("degraded" if chromadb_ok else "unhealthy")

    return {
        "status": overall_status,
        "ollama_healthy": ollama_ok,
        "chromadb_ready": chromadb_ok,
        "documents_count": doc_count,
        "ollama_model": settings.OLLAMA_MODEL,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        "upload_dir": str(settings.UPLOAD_DIR),
        "chroma_dir": str(settings.CHROMA_DIR),
    }
