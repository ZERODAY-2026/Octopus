"""Prompt templates for local LLM-as-judge RAG evaluation metrics.

These mirror the decomposition steps used by the RAGAS framework (Es et al.,
2023, arXiv:2309.15217), adapted so they run against a local Ollama model with
no network access. Every judge prompt demands a terse machine-readable reply so
the parser stays deterministic.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Faithfulness / groundedness and context recall
# ---------------------------------------------------------------------------

CLAIM_EXTRACTION_SYSTEM: str = (
    "You are a strict evaluation annotator for a retrieval-augmented generation "
    "system. You decompose text into atomic, self-contained factual assertions. "
    "You never add information that is not present in the text and you never "
    "comment on the text. You reply with JSON only."
)

CLAIM_EXTRACTION_TEMPLATE: str = """Extract every atomic factual claim from the TEXT below.

Rules:
- One assertion per item; each must stand alone without the surrounding sentence.
- Preserve every number, date, currency amount, and legal period exactly.
- Do not merge two facts into one claim and do not split a single fact in two.
- Do not include opinions, hedges, or meta-commentary.
- If the TEXT contains no factual assertion, return an empty list.
- Return at most {max_claims} claims, preferring the most substantive ones.

TEXT:
\"\"\"
{text}
\"\"\"

Reply with JSON only, in this exact shape:
{{"claims": ["first claim", "second claim"]}}
"""

CLAIM_VERIFICATION_SYSTEM: str = (
    "You are a strict natural-language-inference verifier. You decide whether a "
    "CLAIM is entailed by the EVIDENCE. A claim is supported only when the "
    "evidence states it or follows from it by direct, unambiguous inference. "
    "Plausibility, general legal custom, and outside knowledge do NOT count as "
    "support. If the evidence is silent on the claim, it is not supported. "
    "You reply with a single word."
)

CLAIM_VERIFICATION_TEMPLATE: str = """EVIDENCE:
\"\"\"
{context}
\"\"\"

CLAIM:
\"\"\"
{claim}
\"\"\"

Is the CLAIM entailed by the EVIDENCE? Reply with exactly one word: yes or no.
"""

# ---------------------------------------------------------------------------
# Context relevance (does a retrieved excerpt belong with the question?)
# ---------------------------------------------------------------------------

CONTEXT_RELEVANCE_SYSTEM: str = (
    "You judge whether a retrieved document excerpt is useful for answering a "
    "question. An excerpt is useful if it contains information needed for the "
    "answer, even partially. Topically similar prose that cannot contribute to "
    "the answer is not useful. You reply with a single word."
)

CONTEXT_RELEVANCE_TEMPLATE: str = """QUESTION:
\"\"\"
{question}
\"\"\"

RETRIEVED EXCERPT:
\"\"\"
{context}
\"\"\"

Is this excerpt useful for answering the QUESTION? Reply with exactly one word: yes or no.
"""

# ---------------------------------------------------------------------------
# Answer relevancy (reverse-question generation, as in RAGAS)
# ---------------------------------------------------------------------------

QUESTION_GENERATION_SYSTEM: str = (
    "You write the questions that a given answer would be the perfect response "
    "to. You ignore the answer's wording and reproduce only its information "
    "need. You reply with JSON only."
)

QUESTION_GENERATION_TEMPLATE: str = """Read the ANSWER below and write {num_questions} distinct questions \
that this answer directly and completely satisfies.

Rules:
- Each question must be answerable from the ANSWER alone.
- Do not refer to "the answer", "the passage", or "the document".
- Vary the phrasing between questions.

ANSWER:
\"\"\"
{answer}
\"\"\"

Reply with JSON only, in this exact shape:
{{"questions": ["first question", "second question"]}}
"""

# ---------------------------------------------------------------------------
# Abstention / calibrated uncertainty
# ---------------------------------------------------------------------------

ABSTENTION_SYSTEM: str = (
    "You classify whether an assistant response correctly declines to answer "
    "because the supplied evidence does not contain the requested information. "
    "You reply with a single word."
)

ABSTENTION_TEMPLATE: str = """QUESTION:
\"\"\"
{question}
\"\"\"

ASSISTANT RESPONSE:
\"\"\"
{answer}
\"\"\"

Does the ASSISTANT RESPONSE state that the evidence is silent, unavailable, or \
insufficient rather than asserting an answer? Reply with exactly one word: yes or no.
"""
