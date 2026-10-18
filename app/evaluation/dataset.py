"""Evaluation dataset models and loaders for RAG quality measurement.

Samples mirror the CUAD (Contract Understanding Atticus Dataset) question-answer
shape: a contract, a natural-language question about it, the reference answer a
legal expert would give, and the passages that justify that answer. The
``answerable`` flag drives the calibrated-uncertainty metric, because a triage
agent that invents an answer to an unanswerable question is failing its most
important guarantee.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field

from app.evaluation.text import coverage, normalize_for_match

logger = logging.getLogger(__name__)


class EvaluationSample(BaseModel):
    """A single ground-truth evaluation case for the legal triage pipeline."""

    id: str = Field(..., description="Stable identifier for the sample")
    question: str = Field(..., description="Question asked of the document", min_length=1)
    reference_answer: str = Field(
        default="",
        description="Expert reference answer used for context recall and answer correctness",
    )
    answerable: bool = Field(
        default=True,
        description="False when the document is genuinely silent, so the agent must abstain",
    )
    document: str = Field(
        default="",
        description="Sample document filename this case belongs to",
    )
    relevant_clause_types: list[str] = Field(
        default_factory=list,
        description="Clause categories this case should surface, e.g. ['termination']",
    )
    relevant_keywords: list[str] = Field(
        default_factory=list,
        description="Distinctive phrases used to locate the ground-truth chunks",
    )
    gold_passages: list[str] = Field(
        default_factory=list,
        description="Verbatim source passages that must be cited to answer correctly",
    )
    category: str = Field(
        default="qa",
        description="Sample grouping, e.g. 'qa', 'adversarial', 'silence', 'parsing'",
    )


def load_dataset(path: str | Path) -> list[EvaluationSample]:
    """Load evaluation samples from a ``.jsonl`` or ``.json`` file.

    Args:
        path: Path to a JSONL file (one sample per line) or a JSON file holding
            a list of samples or an object with a ``"samples"`` key.

    Returns:
        List of validated :class:`EvaluationSample` instances.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the file extension is unsupported or the payload is malformed.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    raw_records: list[Any] = []

    if suffix == ".jsonl":
        for line_number, line in enumerate(
            dataset_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                raw_records.append(json.loads(stripped))
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Malformed JSON on line {line_number} of {dataset_path}: {err}"
                ) from err
    elif suffix == ".json":
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_records = payload.get("samples", [])
        elif isinstance(payload, list):
            raw_records = payload
        else:
            raise ValueError(f"Unsupported JSON payload in {dataset_path}")
    else:
        raise ValueError(
            f"Unsupported dataset format '{suffix}'. Use .jsonl or .json."
        )

    samples: list[EvaluationSample] = []
    for record in raw_records:
        if not isinstance(record, dict):
            logger.warning("Skipping non-object evaluation record: %r", record)
            continue
        try:
            samples.append(EvaluationSample(**record))
        except Exception as err:
            logger.warning("Skipping invalid evaluation sample %r: %s", record, err)

    return samples


def filter_by_document(
    samples: Iterable[EvaluationSample], document: str
) -> list[EvaluationSample]:
    """Select the samples that belong to one document.

    Args:
        samples: Candidate samples.
        document: Filename to match against each sample's ``document`` field.

    Returns:
        Samples whose ``document`` matches, or every sample when the field is blank.
    """
    wanted = normalize_for_match(document)
    selected: list[EvaluationSample] = []
    for sample in samples:
        if not sample.document or normalize_for_match(sample.document) == wanted:
            selected.append(sample)
    return selected


def relevant_chunk_ids(
    sample: EvaluationSample,
    chunks: Sequence[Any],
    *,
    fallback_threshold: float = 0.75,
) -> list[str]:
    """Determine which indexed chunks contain this sample's ground-truth evidence.

    Matching is keyword-first because CUAD questions rarely share wording with
    the clause that answers them: an expert question like "what are the
    termination conditions?" is answered by a chunk that never uses the word
    "question". When a sample carries no keywords or gold passages, the
    reference answer is used as a content-term coverage fallback.

    Args:
        sample: The evaluation sample.
        chunks: Chunk-like objects exposing ``chunk_id`` and ``content`` (any
            object with those attributes, or a mapping with those keys).
        fallback_threshold: Minimum reference-answer coverage for the fallback path.

    Returns:
        List of matching chunk identifiers, in the order the chunks were supplied.
    """
    keywords = [normalize_for_match(k) for k in sample.relevant_keywords if k.strip()]
    gold = [normalize_for_match(p) for p in sample.gold_passages if p.strip()]

    matches: list[str] = []
    for chunk in chunks:
        chunk_id = _attr(chunk, "chunk_id")
        content = normalize_for_match(_attr(chunk, "content") or "")
        if not chunk_id or not content:
            continue
        if any(keyword in content for keyword in keywords):
            matches.append(str(chunk_id))
            continue
        if any(passage and passage in content for passage in gold):
            matches.append(str(chunk_id))

    if matches or not sample.reference_answer:
        return matches

    # Fallback: chunks that cover most of the reference answer's content terms.
    for chunk in chunks:
        chunk_id = _attr(chunk, "chunk_id")
        content = _attr(chunk, "content") or ""
        if not chunk_id:
            continue
        if coverage(sample.reference_answer, content, n=1) >= fallback_threshold:
            matches.append(str(chunk_id))
    return matches


def _attr(item: Any, name: str) -> Any:
    """Read a field from either an attribute-carrying object or a mapping.

    Args:
        item: Object or mapping.
        name: Field name.

    Returns:
        The field value, or None when absent.
    """
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)
