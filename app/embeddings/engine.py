"""Embedding engine using SentenceTransformers with local caching and offline execution."""

from __future__ import annotations

import logging
import os
from typing import Any

# Fail closed: never let a library reach the network at run time. Models are
# fetched only by scripts/setup_models.py (which unsets these variables).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import torch
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Computes dense vector representations of legal text using SentenceTransformers."""

    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL,
        device: str | None = None,
    ) -> None:
        """Initialize the embedding model on GPU if available, otherwise CPU.

        Configures local cache usage to prevent unnecessary network requests after download.

        Args:
            model_name: HuggingFace model identifier or local path.
            device: Explicit computation device ('cuda', 'cpu'), or None to auto-detect.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing EmbeddingEngine with model '%s' on device '%s'", self.model_name, self.device)

        try:
            # First attempt loading strictly from local cache to avoid external network requests
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=True,
            )
            logger.info("Loaded embedding model '%s' from local cache", self.model_name)
        except Exception as err:
            raise RuntimeError(
                f"Embedding model '{self.model_name}' is not available locally and the app "
                "never downloads models while running. Run `uv run python scripts/setup_models.py` "
                "once (with network access), then start the app again."
            ) from err

        if hasattr(self.model, "eval"):
            self.model.eval()

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality of the embedding model."""
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
        else:
            dim = getattr(self.model, "get_sentence_embedding_dimension")()
        return int(dim) if dim is not None else 384

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Batch encode a list of text strings into vector representations.

        Args:
            texts: List of text strings to embed.

        Returns:
            A list of float lists representing normalized dense embedding vectors.
        """
        if not texts:
            return []

        embeddings: Any = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_single(self, text: str) -> list[float]:
        """Encode a single text string into an embedding vector.

        Args:
            text: Single text string to embed.

        Returns:
            A list of floats representing the normalized embedding vector.
        """
        results = self.encode([text])
        return results[0]
