"""Calibrated-uncertainty detection.

The project promises that when a document is silent on a question it says so
plainly instead of inventing a clause. That promise is only testable if one
place decides what "saying so" looks like, so this module owns the marker
vocabulary and the matcher, and both the pipeline and the evaluation harness
read it.

The pipeline uses it to decide whether an answer is a refusal — and therefore
whether clause findings raised alongside it should be withheld. The harness uses
it to score abstention recall (did a deliberately unanswerable question get a
refusal?) and its mirror image, over-abstention.

The module is deliberately a leaf: it depends only on the standard library, so
``app.pipeline`` can import it without pulling in the evaluation package.
"""

from __future__ import annotations

import re

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


def is_abstention(text: str) -> bool:
    """Detect whether a response declines to answer for lack of evidence.

    Args:
        text: Assistant answer text.

    Returns:
        True when a calibrated-uncertainty phrase is present.
    """
    normalized = normalize_for_marker_match(text)
    if not normalized:
        return False
    return any(marker in normalized for marker in ABSTENTION_MARKERS)
