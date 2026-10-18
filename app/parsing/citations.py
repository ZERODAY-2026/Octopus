"""Verbatim citation anchoring.

A language model asked to quote a contract does three things reliably: it quotes
approximately, it truncates mid-sentence, and occasionally it puts something that
is not a quotation at all (a refusal, a summary) into the citation field.

The project's core promise is that every plain-language summary maps back to a
*verifiable, quoted passage*. Trusting the model to reproduce those quotes makes
that promise probabilistic. This module makes it structural: a generated quote is
located inside the retrieved source text and replaced by the exact source span,
with the character offsets the source actually uses.

Two outcomes are possible:

* **verbatim** — the quote is found in the source after Unicode, quote-mark and
  whitespace normalization. The emitted text is the source's own characters.
* **repaired** — the quote is a close but imperfect reproduction. The longest
  run of matching tokens is located and emitted as the quotation, so the
  citation is still a true quotation from the document.

A quote that cannot be matched at all yields ``None``; the caller should drop it
and tell the user rather than present an unverifiable passage as a source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Sequence

from app.evaluation.text import normalize_unicode

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Minimum fraction of a quote's tokens that must appear as one unbroken run in
#: the source for the quote to be accepted as a repaired citation.
DEFAULT_MIN_RATIO: float = 0.55

#: Minimum tokens in a matched run. Guards against a two-word coincidence being
#: presented as a quotation.
DEFAULT_MIN_TOKENS: int = 3


@dataclass(frozen=True)
class SourceSpan:
    """A chunk of source text together with its location in the document."""

    content: str
    page_number: int = 1
    start_char: int = 0
    end_char: int = 0

    def __post_init__(self) -> None:
        if not self.end_char:
            object.__setattr__(self, "end_char", self.start_char + len(self.content))


@dataclass(frozen=True)
class CitationMatch:
    """A citation resolved to a real span of the source document."""

    text: str
    page_number: int
    start_char: int
    end_char: int
    match_type: str
    ratio: float = 1.0

    @property
    def repaired(self) -> bool:
        """Whether the model's quote had to be corrected to match the source.

        Returns:
            True when the match was not exact.
        """
        return self.match_type != "verbatim"


def _normalized_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize text to a comparable form while tracking original indices.

    Applies the same folds as :func:`app.evaluation.text.normalize_for_match`
    (Unicode NFKC, typographic quote and dash folding, whitespace collapse,
    lowercasing) but records, for every character of the normalized string, the
    index of the original character it came from. That map is what converts a
    match position back into a real document offset.

    Args:
        text: Original source or quote text.

    Returns:
        Tuple of (normalized text, index map of equal length).
    """
    normalized_chars: list[str] = []
    index_map: list[int] = []
    pending_space = False
    pending_space_index = 0

    folded = normalize_unicode(text)

    for original_index, raw_char in enumerate(folded):
        if raw_char.isspace():
            if not pending_space and normalized_chars:
                pending_space = True
                pending_space_index = original_index
            continue

        if pending_space:
            normalized_chars.append(" ")
            index_map.append(pending_space_index)
            pending_space = False

        lowered = raw_char.lower()
        if len(lowered) != 1:
            lowered = raw_char
        normalized_chars.append(lowered)
        index_map.append(original_index)

    return "".join(normalized_chars), index_map


def _token_spans(normalized: str) -> list[tuple[int, int]]:
    """Find token character ranges inside a normalized string.

    Args:
        normalized: Normalized text.

    Returns:
        List of ``(start, end)`` character offsets, half-open.
    """
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(normalized)]


def _locate_verbatim(quote: str, span: SourceSpan) -> CitationMatch | None:
    """Try to find the quote exactly inside the source span.

    Args:
        quote: Candidate quotation.
        span: Source span to search.

    Returns:
        A verbatim CitationMatch, or None when the quote is not present exactly.
    """
    normalized_quote, _ = _normalized_with_map(quote)
    if not normalized_quote:
        return None

    normalized_source, index_map = _normalized_with_map(span.content)
    position = normalized_source.find(normalized_quote)
    if position < 0:
        return None

    raw_start = index_map[position]
    raw_end = index_map[position + len(normalized_quote) - 1] + 1

    return CitationMatch(
        text=span.content[raw_start:raw_end],
        page_number=span.page_number,
        start_char=span.start_char + raw_start,
        end_char=span.start_char + raw_end,
        match_type="verbatim",
        ratio=1.0,
    )


def _locate_repaired(
    quote: str,
    span: SourceSpan,
    min_ratio: float,
    min_tokens: int,
) -> CiteMatchTuple | None:
    """Locate the longest common run of tokens between a quote and a source span.

    Args:
        quote: Candidate quotation.
        span: Source span to search.
        min_ratio: Minimum fraction of the quote's tokens that must match.
        min_tokens: Minimum length of the matched run.

    Returns:
        A repaired CitationMatch, or None when no run is long enough.
    """
    normalized_quote, _ = _normalized_with_map(quote)
    normalized_source, index_map = _normalized_with_map(span.content)

    quote_tokens = _token_spans(normalized_quote)
    source_tokens = _token_spans(normalized_source)
    if not quote_tokens or not source_tokens:
        return None

    quote_words = [normalized_quote[a:b] for a, b in quote_tokens]
    source_words = [normalized_source[a:b] for a, b in source_tokens]

    matcher = SequenceMatcher(a=source_words, b=quote_words, autojunk=False)
    best = None
    for block in matcher.get_matching_blocks():
        if block.size > 0 and (best is None or block.size > best.size):
            best = block

    if best is None or best.size < min_tokens:
        return None

    ratio = best.size / len(quote_words)
    if ratio < min_ratio:
        return None

    first_source_token = source_tokens[best.a]
    last_source_token = source_tokens[best.a + best.size - 1]
    raw_start = index_map[first_source_token[0]]
    raw_end = index_map[last_source_token[1] - 1] + 1

    return (
        span.content[raw_start:raw_end],
        span.start_char + raw_start,
        span.start_char + raw_end,
        ratio,
    )


CiteMatchTuple = tuple[str, int, int, float]


def snap_citation(
    quote: str,
    spans: Sequence[SourceSpan | tuple[str, int, int, int]],
    *,
    min_ratio: float = DEFAULT_MIN_RATIO,
    min_tokens: int = DEFAULT_MIN_TOKENS,
) -> CitationMatch | None:
    """Resolve a generated quote to an exact span of the source text.

    Verbatim containment is attempted across every span first, so an exact quote
    is never silently rewritten. Only if no exact match exists does the function
    fall back to locating the longest shared token run in the best-matching span.

    Args:
        quote: Quotation produced by the language model.
        spans: Source spans, either :class:`SourceSpan` objects or
            ``(content, page_number, start_char, end_char)`` tuples.
        min_ratio: Minimum fraction of quote tokens for a repaired match.
        min_tokens: Minimum matched run length for a repaired match.

    Returns:
        A CitationMatch whose ``text`` is genuine source text, or None when the
        quote cannot be located in the document at all.
    """
    if not quote or not quote.strip():
        return None

    normalized_spans = [_coerce_span(span) for span in spans]
    if not normalized_spans:
        return None

    for span in normalized_spans:
        match = _locate_verbatim(quote, span)
        if match is not None:
            return match

    best: CitationMatch | None = None
    for span in normalized_spans:
        repaired = _locate_repaired(quote, span, min_ratio, min_tokens)
        if repaired is None:
            continue
        text, start, end, ratio = repaired
        candidate = CitationMatch(
            text=text,
            page_number=span.page_number,
            start_char=start,
            end_char=end,
            match_type="repaired",
            ratio=ratio,
        )
        if best is None or candidate.ratio > best.ratio:
            best = candidate

    return best


def _coerce_span(span: SourceSpan | tuple[str, int, int, int]) -> SourceSpan:
    """Normalize a span argument into a SourceSpan.

    Args:
        span: SourceSpan or a 4-tuple of content, page, start, end.

    Returns:
        A SourceSpan instance.
    """
    if isinstance(span, SourceSpan):
        return span
    content, page_number, start_char, end_char = span
    return SourceSpan(
        content=str(content),
        page_number=int(page_number or 1),
        start_char=int(start_char or 0),
        end_char=int(end_char or 0),
    )


def snap_citations(
    quotes: Iterable[str],
    spans: Sequence[SourceSpan | tuple[str, int, int, int]],
    *,
    min_ratio: float = DEFAULT_MIN_RATIO,
    min_tokens: int = DEFAULT_MIN_TOKENS,
) -> tuple[list[CitationMatch], list[str]]:
    """Resolve many quotes, separating the verified from the unverifiable.

    Args:
        quotes: Candidate quotations.
        spans: Source spans to search.
        min_ratio: Minimum fraction of quote tokens for a repaired match.
        min_tokens: Minimum matched run length for a repaired match.

    Returns:
        Tuple of (matches, rejected_quotes).
    """
    matches: list[CitationMatch] = []
    rejected: list[str] = []

    for quote in quotes:
        match = snap_citation(quote, spans, min_ratio=min_ratio, min_tokens=min_tokens)
        if match is None:
            rejected.append(quote)
        else:
            matches.append(match)

    return matches, rejected


def spans_from_chunks(chunks: Iterable[object]) -> list[SourceSpan]:
    """Build source spans from retrieved chunk objects or mappings.

    Args:
        chunks: Objects exposing ``content``, ``page_number``, ``start_char`` and
            ``end_char``, or mappings with those keys.

    Returns:
        List of SourceSpan instances, skipping entries with empty content.
    """
    spans: list[SourceSpan] = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            content = str(chunk.get("content") or "")
            page = int(chunk.get("page_number") or 1)
            start = int(chunk.get("start_char") or 0)
            end = int(chunk.get("end_char") or 0)
        else:
            content = str(getattr(chunk, "content", "") or "")
            page = int(getattr(chunk, "page_number", 1) or 1)
            start = int(getattr(chunk, "start_char", 0) or 0)
            end = int(getattr(chunk, "end_char", 0) or 0)
        if content.strip():
            spans.append(
                SourceSpan(content=content, page_number=page, start_char=start, end_char=end)
            )
    return spans
