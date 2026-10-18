"""ChromaDB local vector store integration with custom embedding function."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings as ChromaSettings

from app.embeddings.engine import EmbeddingEngine
from app.models import DocumentChunk, RetrievedChunk
from app.retrieval.lexical import BM25Index, expand_query, reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class ChromaEmbeddingAdapter(EmbeddingFunction[Documents]):
    """Adapts application EmbeddingEngine to ChromaDB's EmbeddingFunction interface."""

    def __init__(self, engine: EmbeddingEngine) -> None:
        """Initialize adapter with the given embedding engine.

        Args:
            engine: EmbeddingEngine instance used to compute embeddings.
        """
        super().__init__()
        self.engine = engine

    def __call__(self, input: Documents) -> Embeddings:
        """Embed a list of document strings.

        Args:
            input: List of text documents.

        Returns:
            List of float embedding vectors.
        """
        return self.engine.encode(list(input))


class VectorStore:
    """Local ChromaDB-backed vector store for document chunk storage and similarity retrieval."""

    def __init__(
        self,
        persist_dir: str | Path,
        embedding_engine: EmbeddingEngine,
    ) -> None:
        """Initialize ChromaDB PersistentClient and register the custom embedding function.

        Args:
            persist_dir: Filesystem path where ChromaDB databases are persisted.
            embedding_engine: Pre-initialized EmbeddingEngine.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        # Telemetry is disabled explicitly rather than left to the default.
        # ChromaDB ships an anonymised-telemetry client that is a no-op in the
        # pinned version, but the flag has been an outbound HTTP call in other
        # releases. A zero-network tool must not depend on a dependency's
        # default being quiet, so we state the intent here.
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.embedding_engine = embedding_engine
        self.embedding_function = ChromaEmbeddingAdapter(embedding_engine)
        # BM25 indexes are rebuilt only when a collection's size changes.
        self._bm25_cache: dict[str, tuple[int, BM25Index, list[dict[str, Any]]]] = {}

    @staticmethod
    def _sanitize_collection_name(document_id: str) -> str:
        """Sanitize a document ID into a valid ChromaDB collection name.

        ChromaDB collection constraints:
        - 3 to 63 characters long
        - Alphanumeric start and end
        - Contains only alphanumeric, underscores, or hyphens

        Args:
            document_id: Input identifier.

        Returns:
            A compliant collection name string.
        """
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", document_id)
        if not sanitized or not sanitized[0].isalnum():
            sanitized = f"doc_{sanitized}"
        if not sanitized[-1].isalnum():
            sanitized = f"{sanitized}_doc"
        if len(sanitized) < 3:
            sanitized = (sanitized + "_doc")[:3]
        if len(sanitized) > 63:
            hash_suffix = hashlib.md5(document_id.encode("utf-8")).hexdigest()[:8]
            sanitized = f"{sanitized[:54]}_{hash_suffix}"
        return sanitized

    def add_document(self, document_id: str, chunks: list[DocumentChunk]) -> None:
        """Add chunks for a document into its dedicated ChromaDB collection.

        Metadata (page_number, chunk_index, start_char, end_char, source_file)
        is preserved for citation tracing.

        Args:
            document_id: Unique document identifier used as collection name.
            chunks: List of DocumentChunk instances to insert.
        """
        if not chunks:
            logger.warning("No chunks provided to add for document_id '%s'", document_id)
            return

        collection_name = self._sanitize_collection_name(document_id)
        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"original_document_id": document_id, "hnsw:space": "cosine"},
        )

        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [f"{collection_name}_chunk_{c.chunk_index}_{idx}" for idx, c in enumerate(batch, start=i)]
            documents = [c.content for c in batch]
            metadatas: list[dict[str, Any]] = [
                {
                    "page_number": int(c.page_number),
                    "chunk_index": int(c.chunk_index),
                    "start_char": int(c.start_char),
                    "end_char": int(c.end_char),
                    "source_file": str(c.source_file),
                }
                for c in batch
            ]
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        logger.info(
            "Successfully stored %d chunks for document '%s' in collection '%s'",
            len(chunks),
            document_id,
            collection_name,
        )

    def _lexical_index(self, collection: Any, collection_name: str) -> tuple[BM25Index, list[dict[str, Any]]]:
        """Return (BM25 index, rows) for a collection, cached by chunk count."""
        count = collection.count()
        cached = self._bm25_cache.get(collection_name)
        if cached and cached[0] == count:
            return cached[1], cached[2]
        data = collection.get(include=["documents", "metadatas"])
        rows = [
            {"chunk_id": cid, "content": doc, "metadata": meta or {}}
            for cid, doc, meta in zip(data.get("ids") or [], data.get("documents") or [], data.get("metadatas") or [])
        ]
        index = BM25Index([r["content"] for r in rows])
        self._bm25_cache[collection_name] = (count, index, rows)
        return index, rows

    def query(
        self,
        document_id: str,
        query_text: str,
        n_results: int = 5,
        hybrid: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k chunks using dense + BM25 search fused with RRF.

        The dense leg embeds the raw question; the lexical leg searches the
        question *plus* legal synonyms (see ``expand_query``) so that a question
        such as "termination date" reaches a clause worded "remain in effect
        until terminated ... upon thirty (30) days' prior written notice".

        Args:
            document_id: Target document identifier.
            query_text: Natural language query string.
            n_results: Maximum number of chunks to return.
            hybrid: Disable to get dense-only results (used for ablations).

        Returns:
            List of dictionaries containing 'chunk_id', 'content', 'metadata', 'distance'.
        """
        collection_name = self._sanitize_collection_name(document_id)

        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        except Exception:
            logger.warning("Collection '%s' not found for query", collection_name)
            return []

        total_count = collection.count()
        if total_count == 0:
            return []

        dense_k = min(max(n_results * 4, 20) if hybrid else n_results, total_count)
        raw_results = collection.query(
            query_texts=[query_text],
            n_results=dense_k,
            include=["documents", "metadatas", "distances"],
        )

        dense: list[dict[str, Any]] = []
        ids = (raw_results.get("ids") or [[]])[0]
        docs = (raw_results.get("documents") or [[]])[0]
        metas = (raw_results.get("metadatas") or [[]])[0]
        dists = (raw_results.get("distances") or [[]])[0]

        for position, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            chunk_id = ids[position] if position < len(ids) else f"chunk_{position}"
            dense.append({"chunk_id": chunk_id, "content": doc, "metadata": meta, "distance": dist})

        if not hybrid:
            return dense[:n_results]

        index, rows = self._lexical_index(collection, collection_name)
        lexical_ranked = index.rank(expand_query(query_text), top_n=dense_k)
        by_id: dict[str, dict[str, Any]] = {d["chunk_id"]: d for d in dense}
        lexical_ids: list[str] = []
        for row_idx, _score in lexical_ranked:
            row = rows[row_idx]
            lexical_ids.append(row["chunk_id"])
            by_id.setdefault(row["chunk_id"], {**row, "distance": None})

        fused = reciprocal_rank_fusion([[d["chunk_id"] for d in dense], lexical_ids])
        return [by_id[cid] for cid, _ in fused[:n_results]]

    def query_many(
            self,
            document_ids: list[str],
            query_text: str,
            n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Run ``query`` over several documents; each hit carries its document id.

        Hits are interleaved round-robin so every document is represented
        before any document contributes a second hit.
        """
        per_doc = [
            [{**hit, "document_id": doc_id} for hit in self.query(doc_id, query_text, n_results)]
            for doc_id in document_ids
        ]
        merged: list[dict[str, Any]] = []
        for rank in range(n_results):
            for hits in per_doc:
                if rank < len(hits):
                    merged.append(hits[rank])
        return merged[: max(n_results, len(document_ids))]

    def get_text(self, document_id: str) -> str:
        """Full text of a document (chunks in order); used for name checks."""
        return "\n".join(c.content for c in self.fetch_all(document_id))

    def get_filename(self, document_id: str) -> str:
        """Original filename recorded in chunk metadata, or the id if unknown."""
        chunks = self.fetch_all(document_id)
        return chunks[0].source_file if chunks and chunks[0].source_file else document_id

    def fetch_all(self, document_id: str) -> list[RetrievedChunk]:
        """Fetch every stored chunk for a document, ordered by chunk index.

        Used by the summarizer and by the evaluation harness, which needs the
        full chunk inventory to resolve ground-truth evidence.

        Args:
            document_id: Target document identifier.

        Returns:
            List of RetrievedChunk ordered by ascending chunk index; empty when
            the collection does not exist.
        """
        collection_name = self._sanitize_collection_name(document_id)
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
            data = collection.get(include=["documents", "metadatas"])
        except Exception as err:
            logger.warning("Could not fetch chunks for '%s': %s", document_id, err)
            return []

        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        chunks: list[RetrievedChunk] = []
        for position, (doc, meta) in enumerate(zip(docs, metas)):
            meta = meta or {}
            chunk_id = ids[position] if position < len(ids) else f"chunk_{position}"

            def _as_int(key: str, default: int, _meta: dict[str, Any] = meta) -> int:
                try:
                    return int(_meta.get(key) or default)
                except (TypeError, ValueError):
                    return default

            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    content=str(doc),
                    page_number=max(1, _as_int("page_number", 1)),
                    chunk_index=max(0, _as_int("chunk_index", 0)),
                    start_char=max(0, _as_int("start_char", 0)),
                    end_char=max(0, _as_int("end_char", 0)),
                    source_file=str(meta.get("source_file") or ""),
                    distance=None,
                )
            )

        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    def delete_document(self, document_id: str) -> None:
        """Delete the collection associated with a document.

        Args:
            document_id: Target document identifier.
        """
        collection_name = self._sanitize_collection_name(document_id)
        try:
            self.client.delete_collection(name=collection_name)
            logger.info("Deleted collection '%s'", collection_name)
        except Exception as err:
            logger.warning("Failed to delete collection '%s' (may not exist): %s", collection_name, err)

    def list_documents(self) -> list[str]:
        """List all collection names currently stored in the vector database.

        Returns:
            List of collection name strings.
        """
        collections = self.client.list_collections()
        return [col.name for col in collections]
