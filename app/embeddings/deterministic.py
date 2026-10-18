"""Zero-dependency deterministic embedding fallback.

The production embedding path is ``sentence-transformers`` (see
:mod:`app.embeddings.engine`). That is the right choice for accuracy, but it has
one operational problem for a zero-network tool: the very first run has to
download model weights, which contradicts the privacy guarantee in spirit (and
makes hermetic CI impossible).

:class:`OfflineHashingEmbedder` closes that gap. It is a deterministic
hashing bag-of-words encoder with no model files, no downloads and no network.
It exposes the same ``encode`` / ``encode_single`` / ``dimension`` surface as
:class:`~app.embeddings.engine.EmbeddingEngine`, so it is a drop-in replacement
for:

* hermetic test suites,
* the offline verification script (``scripts/verify_offline.py``),
* evaluation runs that must not depend on model availability.

It is a *lexical* encoder: it matches shared vocabulary and morphology, not
meaning. It is therefore a fallback and a test instrument, never a substitute
for sentence-transformers when answering real legal questions.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

from app.evaluation.text import light_stem, normalize_unicode

#: Default vector width. Large enough to keep collisions rare, small enough to
#: stay fast for thousands of chunks.
DEFAULT_DIMENSION: int = 512


def _hash_feature(feature: str, dimension: int) -> int:
    """Hash a feature string into a bucket index.

    Uses BLAKE2b rather than the built-in salted hash so results are stable
    across processes and interpreter runs, which is what makes the test suite
    reproducible.

    Args:
        feature: Feature string, e.g. ``"terminat"`` or ``"terminat|notice"``.
        dimension: Number of buckets.

    Returns:
        Bucket index in ``[0, dimension)``.
    """
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def features(text: str) -> list[str]:
    """Extract stemmed unigrams and bigrams from text.

    Args:
        text: Input string.

    Returns:
        List of feature strings, with bigrams joined by ``"|"``.
    """
    tokens = [
        light_stem(token)
        for token in normalize_unicode(text).lower().replace("'", "").split()
        if token.strip()
    ]
    # Strip surrounding punctuation so "notice." and "notice" agree.
    cleaned = [token.strip(".,;:()[]\"'`") for token in tokens]
    cleaned = [token for token in cleaned if token]

    unigrams = list(cleaned)
    bigrams = [f"{a}|{b}" for a, b in zip(cleaned, cleaned[1:])]
    return unigrams + bigrams


class OfflineHashingEmbedder:
    """Deterministic, network-free bag-of-words embedder.

    Documents are encoded as L2-normalized term-frequency vectors over hashed
    unigram and bigram buckets, so cosine similarity reflects shared vocabulary
    and phrase overlap.
    """

    #: Marks an embedder whose similarity is lexical rather than semantic.
    #: Metrics that assume a sentence-transformer (notably RAGAS answer
    #: relevancy, which embeds a generated question against the answer) read
    #: this and report themselves as not gate-able instead of silently scoring
    #: a synonym-free answer as irrelevant.
    is_deterministic: bool = True

    def __init__(self, dimension: int = DEFAULT_DIMENSION, bigram_weight: float = 0.5) -> None:
        """Initialize the embedder.

        Args:
            dimension: Vector width.
            bigram_weight: Relative weight applied to bigram features, which
                sharpens phrase-level matches without letting them dominate.
        """
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = int(dimension)
        self.bigram_weight = float(bigram_weight)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into normalized embedding vectors.

        Args:
            texts: Texts to embed.

        Returns:
            List of float vectors of length :attr:`dimension`.
        """
        return [self._encode_one(text) for text in texts]

    def encode_single(self, text: str) -> list[float]:
        """Encode one text into a normalized embedding vector.

        Args:
            text: Text to embed.

        Returns:
            Float vector of length :attr:`dimension`.
        """
        return self._encode_one(text)

    def _encode_one(self, text: str) -> list[float]:
        """Build the hashed term-frequency vector for one text.

        Args:
            text: Text to embed.

        Returns:
            L2-normalized float vector.
        """
        vector = [0.0] * self.dimension
        if not text or not text.strip():
            return vector

        extracted = features(text)
        for feature in extracted:
            weight = self.bigram_weight if "|" in feature else 1.0
            vector[_hash_feature(feature, self.dimension)] += weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    # Alias kept for parity with sentence-transformers' API naming.
    def get_sentence_embedding_dimension(self) -> int:
        """Return the embedding width, matching the SentenceTransformer API.

        Returns:
            Vector dimension.
        """
        return self.dimension


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    """Cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Similarity in ``[-1.0, 1.0]``; 0.0 for degenerate inputs.
    """
    list_a = list(a)
    list_b = list(b)
    if not list_a or len(list_a) != len(list_b):
        return 0.0
    dot = sum(x * y for x, y in zip(list_a, list_b))
    norm_a = math.sqrt(sum(x * x for x in list_a))
    norm_b = math.sqrt(sum(y * y for y in list_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
