"""Question-intent checks: refuse legal verdicts instead of guessing."""

from __future__ import annotations

import re

_VERDICT_RE = re.compile(
    r"\b(is|are|was|would|will|could)\b.{0,40}\b(enforceable|legal|illegal|lawful|unlawful|valid|invalid|void|binding|fair|unfair|safe to sign|legit)\b"
    r"|\b(enforceab|legally binding|hold up in court|stand up in court|will i win|can i sue|should i sign)\b"
    r"|\bcan (they|he|she|the \w+) (legally|lawfully)\b",
    re.IGNORECASE,
)

VERDICT_RESPONSE_PREFIX = (
    "I can't decide whether this agreement or any clause is legally valid, enforceable or fair - "
    "that depends on the governing law and facts outside the document. "
)


def is_verdict_question(question: str) -> bool:
    """True when the question asks for a legal conclusion rather than document content."""
    return bool(_VERDICT_RE.search(question or ""))
