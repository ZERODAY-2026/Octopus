"""Calibrated-uncertainty detection.

The project promises that when a document is silent on a question it says so
plainly instead of inventing a clause. That promise is only testable if one
place decides what "saying so" looks like, so this module owns the marker
vocabulary and the matcher, and both the pipeline and the evaluation harness
read it.

Answers often mix a grounded finding with a silence note ("Shipper pays within
10 days. The contract does not contain a late-fee cap."). A naive substring
match on the whole answer would treat that as a full refusal and wipe verified
clauses. Sentence-level logic keeps partial answers intact:

* :func:`is_abstention` — every substantive sentence is a refusal (full silence)
* :func:`is_partial_abstention` — at least one refusal sentence and at least one
  positive / concrete sentence

The module is deliberately a leaf: it depends only on the standard library, so
``app.pipeline`` can import it without pulling in the evaluation package.
"""

from __future__ import annotations

import re
from typing import Literal

AnswerStatus = Literal["answered", "partial", "silent"]

#: Phrases that mark a calibrated refusal to answer. Every entry is ASCII
#: lower-case, which is why the matcher below only has to fold case and
#: whitespace.
ABSTENTION_MARKERS: tuple[str, ...] = (
    "silent on",
    "is silent",
    "does not address",
    "do not address",
    "not addressed",
    "does not contain",
    "do not contain",
    "not contain",
    "no provision",
    "no explicit provision",
    "does not specify",
    "do not specify",
    "not specified",
    "does not state",
    "not stated",
    "insufficient information",
    "insufficient evidence",
    "cannot be determined",
    "could not be determined",
    "unable to determine",
    "unable to answer",
    "no relevant excerpt",
    "no relevant provision",
    "not found in the document",
    "not present in the",
    "document is empty",
    "outside the scope",
    "no information",
)

_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

#: Tokens that look like concrete document content (durations, parties, money,
#: section references). Presence of any of these makes a sentence "positive"
#: even when an abstention marker also appears nearby in another sentence.
_CONCRETE_CONTENT_RE = re.compile(
    r"("
    r"\b\d+\b|"  # any numeral
    r"\b(days?|months?|years?|hours?|weeks?)\b|"
    r"\b(shall|must|may|will|pays?|terminat\w*|notice|govern\w*|jurisdict\w*)\b|"
    r"\b(party|parties|shipper|transporter|licen[sc]ee|provider|customer)\b|"
    r"\$|§|"
    r"\b(article|section|exhibit|schedule)\b|"
    r"\b(percent|%|interest|fee|cap|liability|indemnit\w*)\b"
    r")",
    re.IGNORECASE,
)

_VALID_STATUSES: frozenset[str] = frozenset({"answered", "partial", "silent"})


def normalize_for_marker_match(text: str) -> str:
    """Reduce text to lowercase with runs of whitespace collapsed.

    Args:
        text: Raw answer text.

    Returns:
        Lower-cased, single-space string, or an empty string for blank input.
    """
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def split_sentences(text: str) -> list[str]:
    """Split answer text into non-empty sentences.

    Args:
        text: Raw answer text.

    Returns:
        List of stripped sentence strings.
    """
    if not text or not text.strip():
        return []
    return [
        fragment.strip()
        for fragment in _SENTENCE_SPLIT_RE.split(text.strip())
        if fragment.strip()
    ]


def sentence_has_abstention_marker(sentence: str) -> bool:
    """Return True when a single sentence contains a calibrated-refusal phrase.

    Args:
        sentence: One sentence of answer text.

    Returns:
        True when any abstention marker is present.
    """
    normalized = normalize_for_marker_match(sentence)
    if not normalized:
        return False
    return any(marker in normalized for marker in ABSTENTION_MARKERS)


def sentence_has_positive_content(sentence: str) -> bool:
    """Return True when a sentence asserts concrete document content.

    A sentence that is only a refusal marker (or empty after stripping) is not
    positive. Concrete content includes numbers, durations, modal obligations,
    party roles, section references, and similar document-grounded tokens.

    Args:
        sentence: One sentence of answer text.

    Returns:
        True when the sentence looks like a substantive finding.
    """
    normalized = normalize_for_marker_match(sentence)
    if not normalized:
        return False
    if sentence_has_abstention_marker(sentence) and not _CONCRETE_CONTENT_RE.search(
        sentence
    ):
        return False
    # Refusal-only sentences that still contain a marker but no concrete tokens
    # were handled above. Remaining text with concrete tokens, or non-marker
    # prose that is not just filler, counts as positive.
    if _CONCRETE_CONTENT_RE.search(sentence):
        return True
    # Non-marker prose of reasonable length without concrete tokens is still a
    # positive statement (e.g. "The parties owe each other a duty of confidence.").
    if not sentence_has_abstention_marker(sentence) and len(normalized.split()) >= 4:
        return True
    return False


def is_abstention(text: str) -> bool:
    """Detect whether a response is a *full* refusal for lack of evidence.

    Returns True only when every substantive sentence is a refusal (matches an
    abstention marker or carries no concrete content). A mixed answer such as
    "X is Y. The excerpts do not specify Z." is **not** a full abstention.

    Args:
        text: Assistant answer text.

    Returns:
        True when the whole answer is a calibrated refusal.
    """
    sentences = split_sentences(text)
    if not sentences:
        return False

    has_any_marker = False
    for sentence in sentences:
        if sentence_has_positive_content(sentence):
            return False
        if sentence_has_abstention_marker(sentence):
            has_any_marker = True
        elif normalize_for_marker_match(sentence):
            # Non-empty sentence with neither positive content nor marker —
            # treat as non-committal filler; does not by itself defeat silence
            # if other sentences are refusals, but alone is not an abstention.
            pass

    return has_any_marker


def is_partial_abstention(text: str) -> bool:
    """Detect a mixed answer: some findings plus an explicit silence note.

    Args:
        text: Assistant answer text.

    Returns:
        True when at least one sentence is positive and at least one carries
        an abstention marker.
    """
    sentences = split_sentences(text)
    if not sentences:
        return False
    has_positive = any(sentence_has_positive_content(s) for s in sentences)
    has_marker = any(sentence_has_abstention_marker(s) for s in sentences)
    return has_positive and has_marker


def resolve_answer_status(
    answer: str,
    model_status: str | None = None,
) -> AnswerStatus:
    """Resolve ``answered`` / ``partial`` / ``silent`` for a reply.

    Prefers a valid model-supplied status. When missing or invalid, falls back
    to sentence-level heuristics. The silent fallback additionally requires the
    abstention marker to appear in the *first* sentence and that the answer
    contain no positive statement — matching the pipeline's conservative
    clause-withholding rule.

    Args:
        answer: Sanitized answer text.
        model_status: Optional status string from the LLM JSON payload.

    Returns:
        One of ``answered``, ``partial``, or ``silent``.
    """
    if model_status is not None:
        normalized = str(model_status).strip().lower()
        if normalized in _VALID_STATUSES:
            return normalized  # type: ignore[return-value]

    sentences = split_sentences(answer)
    if not sentences:
        return "silent"

    if is_partial_abstention(answer):
        return "partial"

    first = sentences[0]
    if (
        is_abstention(answer)
        and sentence_has_abstention_marker(first)
        and not any(sentence_has_positive_content(s) for s in sentences)
    ):
        return "silent"

    if is_abstention(answer):
        return "silent"

    return "answered"
