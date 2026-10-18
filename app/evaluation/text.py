"""Deterministic, fully offline text utilities used by the RAG evaluation metrics.

Everything in this module is pure Python with no model calls and no network
access, so it can run inside the privacy-validation test suite while all
network interfaces are disabled.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------------

# Word tokens plus standalone numeric tokens (including thousands separators).
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']*|\d+(?:[.,]\d+)*")

# Sentence-ish boundaries: terminal punctuation followed by whitespace, or a
# blank line. Semicolons and colons are treated as soft boundaries because
# legal enumerations lean on them heavily.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])|[\r\n]{2,}")

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")

_QUOTE_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2032": "'",
    "\u2033": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    "\u00a0": " ",
    "\u2026": "...",
    "\u202f": " ",
    "\u2009": " ",
}

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

# Deliberately includes legal boilerplate ("shall", "hereof", "party") so that
# overlap scores reflect substance rather than contract-style function words.
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all also am an and any are as at be because
    been before being below between both but by can cannot could did do does doing
    down during each else ever few for from further had has have having he her here
    hereof hereto hereunder hers herself him himself his how however i if in into
    is it its itself just me more most my myself no nor not of off on once only or
    other ought our ours ourselves out over own same she should so some such than
    that the their theirs them themselves then there therefore these they this those
    through to too under until up upon very was we were what when where whether which
    while who whom why will with would you your yours yourself yourselves
    shall may must hereby herein hereinafter hereinbefore hereon heretofore herein
    hereof thereto thereof wherein whereby whereas pursuant provided including
    include includes such said party parties agreement section article clause
    paragraph exhibit schedule annex hereto attached notwithstanding
    """.split()
)

_TENS_WORDS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_SMALL_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_MAGNITUDE_WORDS = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}

_NUMBER_WORDS: frozenset[str] = frozenset(
    set(_SMALL_NUMBER_WORDS) | set(_TENS_WORDS) | set(_MAGNITUDE_WORDS)
)

# Characters that carry no lexical meaning for content matching.
_PUNCT_STRIP_RE = re.compile(r"[^a-z0-9\s]")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_unicode(text: str) -> str:
    """NFKC-normalize text and fold typographic quotes/dashes to ASCII.

    Args:
        text: Arbitrary input string.

    Returns:
        Normalized string with stopword-safe punctuation preserved.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKC", text)
    return "".join(_QUOTE_MAP.get(ch, ch) for ch in decomposed)


def collapse_whitespace(text: str) -> str:
    """Collapse all runs of whitespace (including form feeds) into single spaces.

    Args:
        text: Arbitrary input string.

    Returns:
        Single-space-joined string with no leading or trailing whitespace.
    """
    if not text:
        return ""
    return re.sub(r"[\s\u00a0]+", " ", text).strip()


def normalize_for_match(text: str) -> str:
    """Produce a canonical form used for verbatim citation containment checks.

    The normalization is intentionally conservative: it folds Unicode, unifies
    quotes and dashes, collapses whitespace, and lowercases. It does *not*
    strip punctuation, so a citation still has to be a genuine quotation.

    Args:
        text: Source passage or candidate quotation.

    Returns:
        Canonical lowercase, whitespace-collapsed string.
    """
    return collapse_whitespace(normalize_unicode(text)).lower()


def strip_punctuation(text: str) -> str:
    """Lowercase text and remove every character that is not alphanumeric.

    Args:
        text: Arbitrary input string.

    Returns:
        Lowercase alphanumeric-and-space string.
    """
    return _PUNCT_STRIP_RE.sub(" ", normalize_unicode(text).lower())


# ---------------------------------------------------------------------------
# Tokenization and stemming
# ---------------------------------------------------------------------------


def light_stem(token: str) -> str:
    """Apply a conservative suffix-stripping stem.

    The goal is to make morphological variants of a legal term agree without
    collapsing distinct terms of art. ``termination``/``terminate`` both reduce
    to ``terminat`` and ``obligation``/``obligated`` both reduce to ``obligat``,
    while ``assignment`` and ``assignee`` deliberately stay distinct.

    Args:
        token: Lowercase single token.

    Returns:
        Stemmed token.
    """
    if len(token) <= 3:
        return token

    # Nominalizations first: "-ation" pairs with "-ate" (termination/terminate).
    if token.endswith("ation") and len(token) >= 7:
        return token[:-5] + "at"
    if token.endswith("ate") and len(token) >= 5:
        return token[:-1]
    if token.endswith("ies") and len(token) >= 5:
        return token[:-3] + "y"
    if token.endswith("sses") and len(token) >= 6:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) >= 4:
        return token[:-1]
    if token.endswith("ment") and len(token) >= 6:
        return token[:-4]
    if token.endswith("ing") and len(token) >= 6:
        return token[:-3]
    if token.endswith("ed") and len(token) >= 5:
        return token[:-2]
    return token


def tokenize(text: str, *, content_only: bool = True, stem: bool = True) -> list[str]:
    """Tokenize text into a list of normalized word and number tokens.

    Args:
        text: Input string.
        content_only: When True, drop stopwords and single-character tokens.
        stem: When True, apply light suffix stemming.

    Returns:
        List of tokens in document order.
    """
    if not text:
        return []
    raw = [m.group(0).lower().replace("'", "") for m in _TOKEN_RE.finditer(normalize_unicode(text))]
    tokens: list[str] = []
    for token in raw:
        if content_only and token in STOPWORDS:
            continue
        if content_only and len(token) < 2:
            continue
        tokens.append(light_stem(token) if stem and token.isalpha() else token)
    return tokens


def token_set(text: str, *, content_only: bool = True) -> set[str]:
    """Return the unique content tokens of a string.

    Args:
        text: Input string.
        content_only: When True, drop stopwords.

    Returns:
        Set of tokens.
    """
    return set(tokenize(text, content_only=content_only))


def ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    """Build the set of contiguous n-grams from a token sequence.

    Args:
        tokens: Ordered token sequence.
        n: Gram size; values below 1 return an empty set.

    Returns:
        Set of n-gram tuples.
    """
    if n < 1 or len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def sentences(text: str) -> list[str]:
    """Split text into sentences, dropping empty fragments.

    Args:
        text: Input string.

    Returns:
        List of non-empty sentence strings.
    """
    if not text or not text.strip():
        return []
    parts = _SENTENCE_RE.split(normalize_unicode(text))
    return [p.strip() for p in parts if p and p.strip()]


def extract_numbers(text: str) -> set[str]:
    """Extract every number mentioned in text as a canonical digit string.

    Numeric literals are captured directly, and spelled-out numbers (including
    hyphenated compounds such as "thirty-six") are converted. This is what lets
    the metrics distinguish "30 days" from a hallucinated "90 days".

    Args:
        text: Input string.

    Returns:
        Set of canonical number strings, e.g. ``{"30", "12.5"}``.
    """
    if not text:
        return set()

    normalized = normalize_unicode(text)
    found: set[str] = set()

    for match in _NUMBER_RE.finditer(normalized):
        raw = match.group(0)
        # "1,250" and "1.250,50" are locale noise in contracts; keep the digits.
        found.add(raw.replace(",", ""))

    for word_match in re.finditer(r"[A-Za-z]+(?:-[A-Za-z]+)*", normalized):
        word = word_match.group(0).lower()
        value = _parse_number_word(word)
        if value is not None:
            found.add(str(value))

    return found


def _parse_number_word(word: str) -> int | None:
    """Convert a spelled-out English number word into an integer.

    Handles single words ("thirty"), hyphenated compounds ("thirty-six"), and
    simple magnitude words ("hundred").

    Args:
        word: Lowercase word, possibly hyphenated.

    Returns:
        The integer value, or None when the word is not a number word.
    """
    parts = word.split("-")
    if not all(p in _NUMBER_WORDS for p in parts):
        return None

    total = 0
    current = 0
    for part in parts:
        if part in _SMALL_NUMBER_WORDS:
            current += _SMALL_NUMBER_WORDS[part]
        elif part in _TENS_WORDS:
            current += _TENS_WORDS[part]
        elif part in _MAGNITUDE_WORDS:
            magnitude = _MAGNITUDE_WORDS[part]
            current = (current or 1) * magnitude
            total += current
            current = 0
    total += current
    return total or None


# ---------------------------------------------------------------------------
# Overlap helpers
# ---------------------------------------------------------------------------


def coverage(reference: str, context: str, *, n: int = 1) -> float:
    """Fraction of reference tokens (or n-grams) that appear in the context.

    A value of 1.0 means every token of the reference is present somewhere in
    the context; it says nothing about ordering or truth, which is why the
    claim-verification path exists on top of it.

    Args:
        reference: Text whose tokens must be covered (typically a claim).
        context: Supporting text (typically retrieved excerpts).
        n: Gram size. 1 = unigram token coverage, 2 = bigram coverage.

    Returns:
        Coverage ratio in ``[0.0, 1.0]``; 0.0 when the reference has no content tokens.
    """
    if n <= 1:
        ref = token_set(reference)
        if not ref:
            return 0.0
        return len(ref & token_set(context)) / len(ref)

    ref_grams = ngrams(tokenize(reference), n)
    if not ref_grams:
        # Fall back to unigrams when the reference is shorter than the gram size.
        return coverage(reference, context, n=1)
    return len(ref_grams & ngrams(tokenize(context), n)) / len(ref_grams)


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity between the content-token sets of two strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Similarity in ``[0.0, 1.0]``; 0.0 when either side has no content tokens.
    """
    set_a = token_set(a)
    set_b = token_set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def cosine_counter(a: str, b: str) -> float:
    """Cosine similarity between the term-frequency vectors of two strings.

    Used as a deterministic stand-in for embedding similarity when no embedding
    model is available.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Cosine similarity in ``[0.0, 1.0]``.
    """
    counts_a = Counter(tokenize(a))
    counts_b = Counter(tokenize(b))
    if not counts_a or not counts_b:
        return 0.0
    shared = set(counts_a) & set(counts_b)
    numerator = sum(counts_a[t] * counts_b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in counts_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counts_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numerator / (norm_a * norm_b)


def contains_verbatim(quote: str, source: str) -> bool:
    """Check whether a quotation appears verbatim inside a source document.

    Both sides are canonicalized with :func:`normalize_for_match`, which makes
    the check tolerant of the whitespace and quote-mark differences introduced
    by PDF extraction while still requiring a genuine quotation.

    Args:
        quote: Candidate quotation, typically produced by the LLM.
        source: Full source text or a concatenation of source excerpts.

    Returns:
        True if the normalized quote is contained in the normalized source.
    """
    norm_quote = normalize_for_match(quote)
    norm_source = normalize_for_match(source)
    if not norm_quote or not norm_source:
        return False
    return norm_quote in norm_source


def longest_common_span_ratio(quote: str, source: str) -> float:
    """Length of the longest shared word run, relative to the quote length.

    Complements :func:`contains_verbatim` for near-verbatim citations: it
    reports how much of a quote is an unbroken copy of the source, on a
    ``[0, 1]`` scale. Uses the classic longest-common-substring dynamic program
    over token sequences, keeping only the previous row in memory.

    Args:
        quote: Candidate quotation.
        source: Source text.

    Returns:
        Longest common contiguous token run divided by the quote's token count.
    """
    quote_tokens = tokenize(quote, content_only=False, stem=False)
    source_tokens = tokenize(source, content_only=False, stem=False)
    if not quote_tokens or not source_tokens:
        return 0.0

    previous_row = [0] * (len(source_tokens) + 1)
    best = 0
    for q_token in quote_tokens:
        current_row = [0] * (len(source_tokens) + 1)
        for j, s_token in enumerate(source_tokens, start=1):
            if q_token == s_token:
                current_row[j] = previous_row[j - 1] + 1
                if current_row[j] > best:
                    best = current_row[j]
        previous_row = current_row

    return best / len(quote_tokens)


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean that returns 0.0 for an empty input.

    Args:
        values: Iterable of floats.

    Returns:
        Mean value, or 0.0 when there is nothing to average.
    """
    collected = list(values)
    if not collected:
        return 0.0
    return sum(collected) / len(collected)
