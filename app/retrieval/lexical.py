"""BM25 lexical index, legal query expansion and rank fusion (all local, no network)."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can did do does for from has have how if in is it its of on or "
    "that the their this to was we what when where which who whom why will with you your "
    "me my i there any all under".split()
)

#: Words that name the *kind* of question, not content in the document.
#: Each trigger regex maps to extra search terms that the clause's own wording is likely to use.
QUERY_EXPANSIONS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), terms)
    for pattern, terms in (
        (r"terminat|expir|end date|how long|duration|\bterm\b|last(s)?\b|renew",
         ("term", "terminate", "terminated", "termination", "effective date", "remain in effect",
          "prior written notice", "expire", "renewal", "commence")),
        (r"govern(ing)? law|which law|jurisdiction|venue|where.*dispute|court",
         ("governed by", "laws of the state", "jurisdiction", "venue", "courts", "forum")),
        (r"arbitrat|mediat|dispute",
         ("arbitration", "arbitrator", "arbitral", "mediation", "dispute", "AAA", "JAMS", "ICC")),
        (r"pay|invoice|fee|price|rate|charge|late",
         ("payment", "invoice", "due within", "interest", "charges", "fees", "rates")),
        (r"liab|indemn|damages|cap\b",
         ("liability", "indemnify", "hold harmless", "damages", "consequential", "limitation of liability")),
        (r"confiden|nda|non-disclosure|secret",
         ("confidential", "confidentiality", "disclosure", "survive")),
        (r"assign|transfer|change of control",
         ("assign", "assignment", "transfer", "consent", "successors")),
        (r"force majeure|act of god|beyond.*control",
         ("force majeure", "acts of god", "beyond the reasonable control")),
        (r"notice|notif",
         ("notice", "written notice", "notify", "days")),
        (r"deadline|within|days|due date|time ?limit",
         ("within", "days", "no later than", "prior to", "deadline")),
        (r"penalt|fine|liquidated|breach|default",
         ("penalty", "liquidated damages", "breach", "default", "remedies", "interest")),
        (r"exclusiv|non-?compete|non-?solicit",
         ("exclusive", "non-compete", "non-solicitation", "shall not")),
    )
)


def tokenize(text: str) -> list[str]:
    """Lower-case, stop-word-free tokens with a crude 5-character stem."""
    out = []
    for tok in _TOKEN_RE.findall((text or "").lower()):
        if tok in _STOPWORDS:
            continue
        out.append(tok[:5] if len(tok) > 5 else tok)
    return out


def expand_query(question: str) -> str:
    """Append legal synonyms that the matching clause is likely to contain."""
    extras: list[str] = []
    for pattern, terms in QUERY_EXPANSIONS:
        if pattern.search(question):
            extras.extend(terms)
    if not extras:
        return question
    return f"{question} {' '.join(dict.fromkeys(extras))}"


class BM25Index:
    """Minimal Okapi BM25 over a fixed list of chunk texts."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.tf = [Counter(t) for t in self.doc_tokens]
        df: Counter[str] = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        n = len(documents)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def rank(self, query: str, top_n: int = 20) -> list[tuple[int, float]]:
        """Return ``(doc_index, score)`` pairs, best first, score > 0 only."""
        q_tokens = tokenize(query)
        if not q_tokens or not self.doc_tokens:
            return []
        scores: list[tuple[int, float]] = []
        for i, counts in enumerate(self.tf):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1.0))
            for t in q_tokens:
                f = counts.get(t)
                if f:
                    score += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


def reciprocal_rank_fusion(rankings: Iterable[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse several ranked id lists into one (RRF)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for pos, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + pos)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


_CAP_RUN_RE = re.compile(r"\b(?:[A-Z][\w&.\-]*)(?:\s+(?:[A-Z][\w&.\-]*))+")


def unknown_entity_phrases(question: str, document_text: str) -> list[str]:
    """Capitalised name phrases in the question that do not occur in the document.

    Example: "Energy XXI Ltd" when the parties are "ENERGY XXI GULF COAST, INC.".
    """
    haystack = re.sub(r"\s+", " ", document_text).lower()
    unknown = []
    for match in _CAP_RUN_RE.finditer(question):
        phrase = match.group(0).strip()
        words = phrase.split()
        if words and words[0].lower() in _STOPWORDS | {"what", "which", "who", "when", "is", "the"}:
            phrase = " ".join(words[1:])
        if len(phrase.split()) >= 2 and phrase.lower() not in haystack:
            unknown.append(phrase)
    return unknown
