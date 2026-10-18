"""RAG evaluation metrics for the Contracter legal triage pipeline.

The metric set follows the retrieval-augmented generation evaluation literature
established by RAGAS (Es, James, Espinosa-Anke, Schockaert — *RAGAS: Automated
Evaluation of Retrieval Augmented Generation*, arXiv:2309.15217, EACL 2024) and
the retrieval metrics that surround it: hit rate, recall@k, precision@k, MRR and
nDCG@k.

Two execution modes are supported for every judged metric:

* ``llm``     — the RAGAS decomposition flow, judged by the *local* Ollama model.
                Fully offline, higher fidelity, slower.
* ``lexical`` — a deterministic token/number-overlap approximation. Needs no
                model, so it runs in CI and inside the network-disabled privacy
                tests. It is intentionally conservative: an assertion that
                shares only stopwords with the context does **not** count as
                grounded, because token overlap alone measures copy-paste, not
                faithful reasoning.

Metrics implemented:

===========================  ==========================================
Metric                       Question it answers
===========================  ==========================================
``faithfulness``             Is every claim in the answer grounded in
(groundedness)               the retrieved context?
``answer_relevancy``         Does the answer address the question?
``context_relevance``        Are the retrieved excerpts on-topic?
``context_precision``        Are the relevant excerpts ranked first?
``context_recall``           Did retrieval cover the reference answer?
``answer_correctness``       Precision / recall / F1 against a reference
``retrieval_*``              hit rate, recall@k, precision@k, MRR, nDCG@k
``citation_grounding_rate``  Do cited passages appear verbatim in source?
``citation_locator_accuracy``Are page numbers and offsets correct?
``abstention_detected``      Does the system refuse when the text is
                             silent? (calibrated uncertainty)
===========================  ==========================================
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app.evaluation import prompts as eval_prompts
from app.evaluation.text import (
    collapse_whitespace,
    contains_verbatim,
    cosine_counter,
    coverage,
    extract_numbers,
    longest_common_span_ratio,
    mean,
    ngrams,
    sentences,
    token_set,
    tokenize,
)
from app.parsing.abstention import ABSTENTION_MARKERS as _ABSTENTION_MARKERS
from app.parsing.abstention import is_abstention as _is_abstention
from app.parsing.abstention import is_partial_abstention as _is_partial_abstention
from app.parsing.abstention import resolve_answer_status as _resolve_answer_status

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants (documented so thresholds are auditable, not magic)
# ---------------------------------------------------------------------------

#: Minimum unigram coverage of a claim by the context for the lexical judge to
#: call the claim supported. Set above the function-word floor on purpose.
LEXICAL_SUPPORT_THRESHOLD: float = 0.6

#: Minimum bigram coverage used as a second, stricter gate for long claims.
LEXICAL_SUPPORT_BIGRAM_THRESHOLD: float = 0.34

#: Minimum coverage of a question's content terms for a chunk to count as
#: relevant in the lexical context-relevance judge.
LEXICAL_CHUNK_RELEVANCE_THRESHOLD: float = 0.4

#: Phrases that mark a calibrated refusal to answer. Re-exported from
#: :mod:`app.parsing.abstention` so the pipeline and the harness share exactly
#: one definition.
ABSTENTION_MARKERS: tuple[str, ...] = _ABSTENTION_MARKERS

_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)

JsonDict = dict[str, Any]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricResult:
    """A single metric outcome with its score and supporting evidence.

    Attributes:
        name: Metric identifier, e.g. ``"faithfulness"``.
        score: Normalized score, normally in ``[0.0, 1.0]``.
        mode: ``"lexical"``, ``"llm"``, or ``"deterministic"``.
        details: Machine-readable breakdown (unsupported claims, per-chunk
            verdicts, matched identifiers) used for diagnostics and reporting.
    """

    name: str
    score: float
    mode: str = "deterministic"
    details: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Serialize the result, rounding the score for stable reporting."""
        return {
            "name": self.name,
            "score": round(float(self.score), 6),
            "mode": self.mode,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def is_abstention(text: str) -> bool:
    """Detect whether a response is a *full* refusal for lack of evidence.

    Thin re-export of :func:`app.parsing.abstention.is_abstention`, kept here so
    the evaluation package remains a complete, self-describing surface.

    Args:
        text: Assistant answer text.

    Returns:
        True when the whole answer is a calibrated refusal.
    """
    return _is_abstention(text)


def is_partial_abstention(text: str) -> bool:
    """Detect a mixed answer with both findings and an explicit silence note."""
    return _is_partial_abstention(text)


def resolve_answer_status(answer: str, model_status: str | None = None) -> str:
    """Resolve answered / partial / silent for evaluation scoring."""
    return _resolve_answer_status(answer, model_status=model_status)


def _parse_yes_no(raw: str) -> bool:
    """Parse a judge's one-word verdict, defaulting to "not supported".

    Defaulting to False keeps the metric conservative: an unparseable judge
    reply must never be able to launder an unsupported claim into a supported
    one.

    Args:
        raw: Raw judge output.

    Returns:
        True only for an unambiguous affirmative.
    """
    if not raw:
        return False
    stripped = raw.strip().lower()
    if stripped.startswith("yes"):
        return True
    if stripped.startswith("no"):
        return False
    # Look for a bare verdict anywhere in a chatty reply.
    has_yes = bool(_YES_RE.search(stripped))
    has_no = bool(_NO_RE.search(stripped))
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return False


def _join_contexts(contexts: Sequence[str]) -> str:
    """Concatenate retrieved excerpts into one evidence string.

    Args:
        contexts: Retrieved chunk texts.

    Returns:
        Double-newline-joined evidence text.
    """
    return "\n\n".join(c for c in contexts if c and c.strip())


def _extract_json_list(raw: str, key: str) -> list[str]:
    """Pull a list of strings out of a loose JSON-ish judge response.

    Args:
        raw: Raw judge output, possibly markdown-fenced or chatty.
        key: JSON key holding the list, e.g. ``"claims"``.

    Returns:
        List of cleaned strings; empty when nothing usable was found.
    """
    import json

    if not raw or not raw.strip():
        return []

    candidates: list[str] = [raw.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    brace_start, brace_end = raw.find("{"), raw.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidates.append(raw[brace_start : brace_end + 1])
    bracket_start, bracket_end = raw.find("["), raw.rfind("]")
    if bracket_start != -1 and bracket_end > bracket_start:
        candidates.append(raw[bracket_start : bracket_end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        values: Any = None
        if isinstance(data, dict):
            values = data.get(key)
        elif isinstance(data, list):
            values = data
        if isinstance(values, list):
            cleaned = [str(v).strip() for v in values if str(v).strip()]
            if cleaned:
                return cleaned

    # Last resort: numbered or bulleted lines.
    lines = [
        re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        for line in raw.splitlines()
    ]
    return [line for line in lines if len(line) > 3]


# ---------------------------------------------------------------------------
# LLM-as-judge client
# ---------------------------------------------------------------------------


class LLMJudge:
    """Local, Ollama-backed judge implementing the decomposition steps of RAGAS.

    Every call goes to the configured local Ollama endpoint. Nothing is sent
    off-device, which keeps the evaluation harness inside the project's
    zero-network privacy guarantee.
    """

    def __init__(self, llm_client: Any, temperature: float = 0.0) -> None:
        """Initialize the judge.

        Args:
            llm_client: An ``app.llm.inference.LLMClient``-compatible object
                exposing ``async generate(prompt, system_prompt, temperature)``.
            temperature: Sampling temperature; 0.0 for reproducible judging.
        """
        self.llm_client = llm_client
        self.temperature = temperature

    async def _ask(self, prompt: str, system_prompt: str) -> str:
        """Send one judge prompt to the local model.

        Args:
            prompt: User-role prompt body.
            system_prompt: Judge role instructions.

        Returns:
            Model response text, or an empty string on failure.
        """
        try:
            return await self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
            )
        except Exception as err:  # pragma: no cover - transport failures
            logger.warning("LLM judge call failed (%s); treating as unsupported", err)
            return ""

    async def extract_claims(self, text: str, max_claims: int = 12) -> list[str]:
        """Decompose text into atomic factual claims.

        Args:
            text: Answer or reference answer to decompose.
            max_claims: Upper bound on returned claims.

        Returns:
            List of claim strings; empty when the model returns nothing usable.
        """
        if not text or not text.strip():
            return []
        prompt = eval_prompts.CLAIM_EXTRACTION_TEMPLATE.format(
            text=collapse_whitespace(text)[:6000],
            max_claims=max_claims,
        )
        raw = await self._ask(prompt, eval_prompts.CLAIM_EXTRACTION_SYSTEM)
        return _extract_json_list(raw, "claims")

    async def verify_claim(self, claim: str, context: str) -> bool:
        """Decide whether the evidence entails a claim.

        Args:
            claim: Atomic claim to verify.
            context: Concatenated retrieved evidence.

        Returns:
            True when the judge reports entailment.
        """
        if not claim.strip() or not context.strip():
            return False
        prompt = eval_prompts.CLAIM_VERIFICATION_TEMPLATE.format(
            context=collapse_whitespace(context)[:6000],
            claim=claim.strip(),
        )
        raw = await self._ask(prompt, eval_prompts.CLAIM_VERIFICATION_SYSTEM)
        return _parse_yes_no(raw)

    async def judge_context_useful(self, question: str, context: str) -> bool:
        """Decide whether one retrieved excerpt helps answer the question.

        Args:
            question: User question.
            context: A single retrieved excerpt.

        Returns:
            True when the judge reports the excerpt is useful.
        """
        if not context.strip():
            return False
        prompt = eval_prompts.CONTEXT_RELEVANCE_TEMPLATE.format(
            question=question.strip(),
            context=collapse_whitespace(context)[:4000],
        )
        raw = await self._ask(prompt, eval_prompts.CONTEXT_RELEVANCE_SYSTEM)
        return _parse_yes_no(raw)

    async def generate_questions(self, answer: str, num_questions: int = 3) -> list[str]:
        """Generate the questions an answer would satisfy.

        Args:
            answer: Generated answer.
            num_questions: Number of questions to request.

        Returns:
            List of generated questions.
        """
        if not answer or not answer.strip():
            return []
        prompt = eval_prompts.QUESTION_GENERATION_TEMPLATE.format(
            answer=collapse_whitespace(answer)[:4000],
            num_questions=num_questions,
        )
        raw = await self._ask(prompt, eval_prompts.QUESTION_GENERATION_SYSTEM)
        return _extract_json_list(raw, "questions")[:num_questions]

    async def judge_abstention(self, question: str, answer: str) -> bool:
        """Decide whether a response correctly declines to answer.

        Args:
            question: User question.
            answer: Assistant response.

        Returns:
            True when the judge reports a calibrated refusal.
        """
        prompt = eval_prompts.ABSTENTION_TEMPLATE.format(
            question=question.strip(),
            answer=collapse_whitespace(answer)[:3000],
        )
        raw = await self._ask(prompt, eval_prompts.ABSTENTION_SYSTEM)
        return _parse_yes_no(raw)


# ---------------------------------------------------------------------------
# Deterministic retrieval metrics
# ---------------------------------------------------------------------------


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant.

    Args:
        retrieved_ids: Retrieved identifiers, best first.
        relevant_ids: Ground-truth relevant identifiers.
        k: Cutoff. Values below 1 return 0.0.

    Returns:
        Precision@k in ``[0.0, 1.0]``.
    """
    if k < 1:
        return 0.0
    top = list(retrieved_ids[:k])
    if not top:
        return 0.0
    relevant = set(relevant_ids)
    return sum(1 for doc_id in top if doc_id in relevant) / len(top)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of all relevant items found within the top-k results.

    Args:
        retrieved_ids: Retrieved identifiers, best first.
        relevant_ids: Ground-truth relevant identifiers.
        k: Cutoff.

    Returns:
        Recall@k in ``[0.0, 1.0]``; 1.0 when there are no relevant items to find.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0
    top = set(list(retrieved_ids[:k])) if k >= 1 else set()
    return len(top & relevant) / len(relevant)


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Whether at least one relevant item appears in the top-k results.

    Args:
        retrieved_ids: Retrieved identifiers, best first.
        relevant_ids: Ground-truth relevant identifiers.
        k: Cutoff.

    Returns:
        1.0 if any relevant item was retrieved, else 0.0.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0
    return 1.0 if set(list(retrieved_ids[:k])) & relevant else 0.0


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant retrieved item.

    Args:
        retrieved_ids: Retrieved identifiers, best first.
        relevant_ids: Ground-truth relevant identifiers.

    Returns:
        ``1/rank`` of the first hit, or 0.0 when nothing relevant was retrieved.
    """
    relevant = set(relevant_ids)
    for position, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Binary-gain normalized discounted cumulative gain at k.

    Args:
        retrieved_ids: Retrieved identifiers, best first.
        relevant_ids: Ground-truth relevant identifiers.
        k: Cutoff.

    Returns:
        nDCG@k in ``[0.0, 1.0]``.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0
    gains = [1.0 if doc_id in relevant else 0.0 for doc_id in list(retrieved_ids[:k])]
    dcg = sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))
    ideal = sum(
        1.0 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1)
    )
    if ideal == 0:
        return 0.0
    return min(1.0, dcg / ideal)


# ---------------------------------------------------------------------------
# Lexical claim support
# ---------------------------------------------------------------------------


def _numbers_grounded(claim: str, context: str) -> tuple[bool, list[str]]:
    """Check that every number in a claim also appears in the context.

    This is the single most valuable guard for legal triage: notice periods,
    penalty amounts and liability caps are numeric, and a wrong number is the
    classic hallucination.

    Args:
        claim: Claim text.
        context: Evidence text.

    Returns:
        Tuple of (all grounded, list of unsupported number strings).
    """
    claim_numbers = extract_numbers(claim)
    if not claim_numbers:
        return True, []
    context_numbers = extract_numbers(context)
    unsupported = sorted(claim_numbers - context_numbers)
    return not unsupported, unsupported


def _lexical_claim_supported(claim: str, context: str) -> bool:
    """Decide claim support using overlapping content tokens and numbers.

    Args:
        claim: Claim text.
        context: Evidence text.

    Returns:
        True when coverage *and* numeric checks both pass.
    """
    if not claim.strip():
        return False
    numbers_ok, _ = _numbers_grounded(claim, context)
    if not numbers_ok:
        return False
    unigram = coverage(claim, context, n=1)
    if unigram < LEXICAL_SUPPORT_THRESHOLD:
        return False
    claim_tokens = tokenize(claim)
    if len(claim_tokens) >= 8:
        return coverage(claim, context, n=2) >= LEXICAL_SUPPORT_BIGRAM_THRESHOLD
    return True


# ---------------------------------------------------------------------------
# Faithfulness / groundedness
# ---------------------------------------------------------------------------


async def faithfulness(
    answer: str,
    contexts: Sequence[str],
    judge: LLMJudge | None = None,
    *,
    max_claims: int = 12,
) -> MetricResult:
    """Measure whether every claim in the answer is grounded in the context.

    This is RAGAS *faithfulness* (a.k.a. groundedness): the answer is
    decomposed into atomic claims and each is verified against the retrieved
    evidence. A calibrated refusal has no claims to verify and scores 1.0.

    Args:
        answer: Generated answer.
        contexts: Retrieved excerpts used to produce the answer.
        judge: Optional LLM judge. When omitted, the deterministic lexical judge
            is used.
        max_claims: Cap on the number of claims verified.

    Returns:
        MetricResult named ``faithfulness``; ``details`` lists the unsupported
        claims so a failing run is actionable.
    """
    evidence = _join_contexts(contexts)
    if not answer or not answer.strip():
        return MetricResult("faithfulness", 0.0, "lexical", {"reason": "empty_answer"})

    if not contexts:
        return MetricResult(
            "faithfulness",
            0.0,
            "lexical",
            {"reason": "no_context_provided", "unsupported_claims": [answer.strip()]},
        )

    if judge is None:
        # A sentence that merely reports silence asserts nothing about the
        # document, so it is not a claim to verify. Removing it here keeps a
        # calibrated refusal at full faithfulness while a refusal that also
        # sneaks in a real assertion is still checked on that assertion.
        claims = [
            sentence
            for sentence in sentences(answer)
            if token_set(sentence) and not is_abstention(sentence)
        ]
        mode = "lexical"
    else:
        claims = await judge.extract_claims(answer, max_claims=max_claims)
        mode = "llm"

    if not claims:
        # Nothing factual asserted: a refusal is maximally faithful.
        ab = is_abstention(answer)
        return MetricResult(
            "faithfulness",
            1.0 if ab else 0.0,
            mode,
            {"no_claims": True, "abstention": ab},
        )

    supported: list[str] = []
    unsupported: list[str] = []
    for claim in claims:
        if judge is None:
            ok = _lexical_claim_supported(claim, evidence)
        else:
            ok = await judge.verify_claim(claim, evidence)
        (supported if ok else unsupported).append(claim)

    score = len(supported) / len(claims)
    return MetricResult(
        "faithfulness",
        score,
        mode,
        {
            "total_claims": len(claims),
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "hallucination_rate": round(1.0 - score, 6),
            "abstention": is_abstention(answer),
        },
    )


# ---------------------------------------------------------------------------
# Answer relevancy
# ---------------------------------------------------------------------------


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two dense vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in ``[-1.0, 1.0]``; 0.0 for degenerate inputs.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def answer_relevancy(
    question: str,
    answer: str,
    judge: LLMJudge | None = None,
    embedder: Any | None = None,
    *,
    num_questions: int = 3,
) -> MetricResult:
    """Measure whether the answer actually addresses the question.

    With a judge and an embedder this reproduces the RAGAS method: generate
    ``num_questions`` questions from the answer, embed them alongside the
    original question, and average the cosine similarities. Without an embedder
    (or without a judge) it falls back to a deterministic lexical blend of
    question-term coverage and cosine term-frequency similarity.

    A non-committal answer is penalized even when it shares vocabulary with the
    question, because silence is not relevance.

    Args:
        question: User question.
        answer: Generated answer.
        judge: Optional LLM judge for reverse-question generation.
        embedder: Optional embedding engine exposing ``encode(list[str])``.
        num_questions: Number of reverse questions to generate.

    Returns:
        MetricResult named ``answer_relevancy``.
    """
    if not answer or not answer.strip():
        return MetricResult("answer_relevancy", 0.0, "lexical", {"reason": "empty_answer"})
    if not question or not question.strip():
        return MetricResult("answer_relevancy", 0.0, "lexical", {"reason": "empty_question"})

    ab = is_abstention(answer)

    if judge is not None and embedder is not None:
        generated = await judge.generate_questions(answer, num_questions=num_questions)
        if generated:
            try:
                vectors = embedder.encode([question, *generated])
                q_vec = vectors[0]
                similarities = [_cosine(q_vec, v) for v in vectors[1:]]
                score = max(0.0, mean(similarities))
                return MetricResult(
                    "answer_relevancy",
                    score,
                    "llm",
                    {
                        "generated_questions": generated,
                        "similarities": [round(s, 6) for s in similarities],
                        "abstention": ab,
                    },
                )
            except Exception as err:  # pragma: no cover - embedding failures
                logger.warning("Embedding failed for answer relevancy (%s); using lexical", err)

    if judge is not None:
        generated = await judge.generate_questions(answer, num_questions=num_questions)
        if generated:
            # Judge available but no embedder: score by lexical similarity of the
            # generated questions to the original one.
            score = max(
                0.0,
                mean(max(cosine_counter(question, q), coverage(q, question)) for q in generated),
            )
            return MetricResult(
                "answer_relevancy",
                score,
                "llm",
                {"generated_questions": generated, "embedder": "unavailable", "abstention": ab},
            )

    question_coverage = coverage(question, answer, n=1)
    cosine = cosine_counter(question, answer)
    score = 0.5 * question_coverage + 0.5 * cosine
    if ab:
        # A refusal still has to admit the topic; cap it so silence never looks relevant.
        score = min(score, 0.35)
    return MetricResult(
        "answer_relevancy",
        max(0.0, min(1.0, score)),
        "lexical",
        {
            "question_coverage": round(question_coverage, 6),
            "cosine": round(cosine, 6),
            "abstention": ab,
        },
    )


# ---------------------------------------------------------------------------
# Context relevance / precision / recall
# ---------------------------------------------------------------------------


async def context_relevance(
    question: str,
    contexts: Sequence[str],
    judge: LLMJudge | None = None,
) -> MetricResult:
    """Measure the proportion of retrieved excerpts that help answer the question.

    This is the chunk-level counterpart of RAGAS context relevance and is the
    metric that isolates *retrieval* failures from *generation* failures.

    Args:
        question: User question.
        contexts: Retrieved excerpts, best first.
        judge: Optional LLM judge.

    Returns:
        MetricResult named ``context_relevance``.
    """
    if not contexts:
        return MetricResult("context_relevance", 0.0, "lexical", {"reason": "no_context"})

    verdicts: list[bool] = []
    for context in contexts:
        if judge is None:
            verdicts.append(
                coverage(question, context, n=1) >= LEXICAL_CHUNK_RELEVANCE_THRESHOLD
            )
        else:
            verdicts.append(await judge.judge_context_useful(question, context))

    score = sum(1 for v in verdicts if v) / len(verdicts)
    return MetricResult(
        "context_relevance",
        score,
        "lexical" if judge is None else "llm",
        {"per_chunk_relevant": verdicts, "num_chunks": len(contexts)},
    )


async def context_precision(
    question: str,
    contexts: Sequence[str],
    judge: LLMJudge | None = None,
) -> MetricResult:
    """Rank-weighted context precision (RAGAS v0.1 formulation).

    Relevant excerpts contribute more when they appear early, so a retriever
    that buries the useful clause at rank 5 scores below one that returns it
    first.

    Args:
        question: User question.
        contexts: Retrieved excerpts, best first.
        judge: Optional LLM judge.

    Returns:
        MetricResult named ``context_precision`` in ``[0.0, 1.0]``.
    """
    if not contexts:
        return MetricResult("context_precision", 0.0, "lexical", {"reason": "no_context"})

    relevances: list[bool] = []
    for context in contexts:
        if judge is None:
            relevances.append(
                coverage(question, context, n=1) >= LEXICAL_CHUNK_RELEVANCE_THRESHOLD
            )
        else:
            relevances.append(await judge.judge_context_useful(question, context))

    denominator = sum(1 for r in relevances if r)
    if denominator == 0:
        return MetricResult(
            "context_precision",
            0.0,
            "lexical" if judge is None else "llm",
            {"per_chunk_relevant": relevances, "reason": "no_relevant_chunks"},
        )

    numerator = 0.0
    running_relevant = 0
    for position, relevant in enumerate(relevances, start=1):
        if relevant:
            running_relevant += 1
            numerator += (running_relevant / position)

    score = numerator / denominator
    return MetricResult(
        "context_precision",
        score,
        "lexical" if judge is None else "llm",
        {"per_chunk_relevant": relevances, "num_relevant": denominator},
    )


async def context_recall(
    reference: str,
    contexts: Sequence[str],
    judge: LLMJudge | None = None,
    *,
    max_claims: int = 12,
) -> MetricResult:
    """Measure how much of the reference answer the retrieved context supports.

    This is RAGAS *context recall*: the reference answer is decomposed into
    claims and each claim is checked against the retrieved evidence. A low score
    means retrieval is missing the clauses needed for a complete answer.

    Args:
        reference: Ground-truth reference answer.
        contexts: Retrieved excerpts.
        judge: Optional LLM judge.
        max_claims: Cap on claims verified.

    Returns:
        MetricResult named ``context_recall``.
    """
    evidence = _join_contexts(contexts)
    if not reference or not reference.strip():
        return MetricResult("context_recall", 1.0, "lexical", {"reason": "no_reference"})
    if not evidence:
        return MetricResult("context_recall", 0.0, "lexical", {"reason": "no_context"})

    if judge is None:
        claims = [s for s in sentences(reference) if token_set(s)]
        mode = "lexical"
    else:
        claims = await judge.extract_claims(reference, max_claims=max_claims)
        mode = "llm"

    if not claims:
        return MetricResult("context_recall", 0.0, mode, {"no_claims": True})

    attributable: list[str] = []
    missing: list[str] = []
    for claim in claims:
        if judge is None:
            ok = _lexical_claim_supported(claim, evidence)
        else:
            ok = await judge.verify_claim(claim, evidence)
        (attributable if ok else missing).append(claim)

    score = len(attributable) / len(claims)
    return MetricResult(
        "context_recall",
        score,
        mode,
        {
            "total_claims": len(claims),
            "attributable_claims": attributable,
            "missing_claims": missing,
        },
    )


# ---------------------------------------------------------------------------
# Answer correctness (precision / recall / F1)
# ---------------------------------------------------------------------------


def token_precision_recall_f1(prediction: str, reference: str) -> tuple[float, float, float]:
    """Token-level precision, recall and F1 over content tokens.

    Args:
        prediction: Generated answer.
        reference: Reference answer.

    Returns:
        Tuple of ``(precision, recall, f1)``; all zeros when either side is empty.
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    pred_counter: dict[str, int] = {}
    for token in pred_tokens:
        pred_counter[token] = pred_counter.get(token, 0) + 1
    ref_counter: dict[str, int] = {}
    for token in ref_tokens:
        ref_counter[token] = ref_counter.get(token, 0) + 1

    overlap = sum(
        min(count, ref_counter.get(token, 0)) for token, count in pred_counter.items()
    )
    if overlap == 0:
        return 0.0, 0.0, 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


async def answer_correctness(
    answer: str,
    reference: str,
    judge: LLMJudge | None = None,
) -> MetricResult:
    """Measure agreement between the answer and a reference answer.

    Combines token-level precision/recall/F1 with bigram F1, which is far more
    sensitive than unigram overlap to word-order hallucinations (for example
    "party A indemnifies party B" versus the reverse).

    Args:
        answer: Generated answer.
        reference: Ground-truth reference answer.
        judge: Optional LLM judge. When supplied, agreement uses
            claim-level precision/recall/F1 for a sharper signal.

    Returns:
        MetricResult named ``answer_correctness`` whose ``details`` carry
        ``precision``, ``recall`` and ``f1``.
    """
    if not answer or not answer.strip():
        return MetricResult(
            "answer_correctness",
            0.0,
            "lexical",
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "reason": "empty_answer"},
        )
    if not reference or not reference.strip():
        return MetricResult(
            "answer_correctness",
            0.0,
            "lexical",
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "reason": "no_reference"},
        )

    if judge is not None:
        pred_claims = await judge.extract_claims(answer)
        ref_claims = await judge.extract_claims(reference)
        if pred_claims and ref_claims:
            # A predicted claim counts as correct when it entails at least one
            # reference claim (verification is done against the reference text).
            reference_text = "\n".join(ref_claims)
            true_positives = 0
            for claim in pred_claims:
                if await judge.verify_claim(claim, reference_text):
                    true_positives += 1
            false_positives = len(pred_claims) - true_positives
            answer_text = "\n".join(pred_claims)
            missed = 0
            for claim in ref_claims:
                if not await judge.verify_claim(claim, answer_text):
                    missed += 1
            false_negatives = missed

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives)
                else 0.0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives)
                else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            return MetricResult(
                "answer_correctness",
                f1,
                "llm",
                {
                    "precision": round(precision, 6),
                    "recall": round(recall, 6),
                    "f1": round(f1, 6),
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                },
            )

    precision, recall, f1 = token_precision_recall_f1(answer, reference)
    gram_precision, gram_recall, gram_f1 = _ngram_precision_recall_f1(answer, reference, n=2)
    combined_f1 = 0.5 * f1 + 0.5 * gram_f1
    return MetricResult(
        "answer_correctness",
        combined_f1,
        "lexical",
        {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "bigram_precision": round(gram_precision, 6),
            "bigram_recall": round(gram_recall, 6),
            "bigram_f1": round(gram_f1, 6),
        },
    )


def _ngram_precision_recall_f1(
    prediction: str, reference: str, n: int
) -> tuple[float, float, float]:
    """Precision, recall and F1 over n-gram sets.

    Args:
        prediction: Generated answer.
        reference: Reference answer.
        n: Gram size.

    Returns:
        Tuple of ``(precision, recall, f1)``.
    """
    pred_grams = ngrams(tokenize(prediction), n)
    ref_grams = ngrams(tokenize(reference), n)
    if not pred_grams or not ref_grams:
        return 0.0, 0.0, 0.0
    overlap = len(pred_grams & ref_grams)
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(pred_grams)
    recall = overlap / len(ref_grams)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Citation verification (project-specific)
# ---------------------------------------------------------------------------


def citation_grounding_rate(
    cited_passages: Sequence[str],
    source_texts: Sequence[str],
    *,
    near_verbatim_threshold: float = 0.85,
) -> MetricResult:
    """Check that every cited passage really appears in the source documents.

    This is the guard for the project's core promise: every plain-language
    summary maps back to a verifiable quotation. A citation is either found
    verbatim (after whitespace/quote normalization) or accepted as near-verbatim
    when at least ``near_verbatim_threshold`` of its word sequence is an
    unbroken run from the source.

    Args:
        cited_passages: Quotation strings produced by the LLM.
        source_texts: Source document text(s) the quotes must come from.
        near_verbatim_threshold: Minimum longest-common-span ratio to accept a
            non-exact quotation.

    Returns:
        MetricResult named ``citation_grounding_rate``; ``details`` records the
        verdict and match type for every citation.
    """
    if not cited_passages:
        return MetricResult(
            "citation_grounding_rate",
            1.0,
            "deterministic",
            {"reason": "no_citations", "citations": []},
        )
    if not source_texts:
        return MetricResult(
            "citation_grounding_rate",
            0.0,
            "deterministic",
            {"reason": "no_source", "citations": list(cited_passages)},
        )

    combined_source = _join_contexts(list(source_texts))
    verdicts: list[JsonDict] = []
    grounded = 0
    for quote in cited_passages:
        if not quote or not quote.strip():
            verdicts.append({"quote": quote, "grounded": False, "match": "empty"})
            continue
        if contains_verbatim(quote, combined_source):
            verdicts.append({"quote": quote, "grounded": True, "match": "verbatim"})
            grounded += 1
            continue
        span_ratio = longest_common_span_ratio(quote, combined_source)
        near = span_ratio >= near_verbatim_threshold
        verdicts.append(
            {
                "quote": quote,
                "grounded": near,
                "match": "near_verbatim" if near else "not_found",
                "longest_common_span_ratio": round(span_ratio, 6),
            }
        )
        if near:
            grounded += 1

    return MetricResult(
        "citation_grounding_rate",
        grounded / len(cited_passages),
        "deterministic",
        {"citations": verdicts, "num_citations": len(cited_passages)},
    )


def citation_locator_accuracy(
    citations: Sequence[JsonDict],
    source_pages: Sequence[JsonDict],
) -> MetricResult:
    """Verify that citation page numbers and character offsets point at the quote.

    Args:
        citations: Dictionaries with ``text``, ``page_number``, ``start_char``
            and ``end_char`` keys, as produced by the triage pipeline.
        source_pages: Page dictionaries with ``page_number`` and ``text`` keys,
            as produced by ``app.parsing.extractor``.

    Returns:
        MetricResult named ``citation_locator_accuracy``.
    """
    if not citations:
        return MetricResult(
            "citation_locator_accuracy",
            1.0,
            "deterministic",
            {"reason": "no_citations", "citations": []},
        )

    pages = {
        int(page.get("page_number", 0)): str(page.get("text", ""))
        for page in source_pages
    }

    verdicts: list[JsonDict] = []
    correct = 0
    for citation in citations:
        text = str(citation.get("text", ""))
        page_number = int(citation.get("page_number", 0) or 0)
        page_text = pages.get(page_number)
        if page_text is None:
            verdicts.append(
                {"text": text, "page_number": page_number, "locator_correct": False,
                 "reason": "page_not_found"}
            )
            continue
        if contains_verbatim(text, page_text):
            correct += 1
            verdicts.append(
                {"text": text, "page_number": page_number, "locator_correct": True,
                 "reason": "found_on_cited_page"}
            )
            continue
        # Check whether the quote exists anywhere else, which pinpoints an
        # off-by-one page attribution rather than a fabricated quote.
        elsewhere = [p for p, t in pages.items() if p != page_number and contains_verbatim(text, t)]
        verdicts.append(
            {
                "text": text,
                "page_number": page_number,
                "locator_correct": False,
                "reason": "wrong_page" if elsewhere else "not_on_any_page",
                "actual_pages": elsewhere,
            }
        )

    return MetricResult(
        "citation_locator_accuracy",
        correct / len(citations),
        "deterministic",
        {"citations": verdicts, "num_citations": len(citations)},
    )


def citation_offset_accuracy(
    citations: Sequence[JsonDict],
    source_pages: Sequence[JsonDict],
) -> MetricResult:
    """Check that ``start_char``/``end_char`` ranges actually bound the quoted text.

    Args:
        citations: Citation dictionaries including offset fields.
        source_pages: Page dictionaries from the extractor.

    Returns:
        MetricResult named ``citation_offset_accuracy``.
    """
    if not citations:
        return MetricResult(
            "citation_offset_accuracy",
            1.0,
            "deterministic",
            {"reason": "no_citations", "citations": []},
        )

    pages = {
        int(page.get("page_number", 0)): str(page.get("text", ""))
        for page in source_pages
    }

    verdicts: list[JsonDict] = []
    correct = 0
    considered = 0
    for citation in citations:
        text = str(citation.get("text", ""))
        start = int(citation.get("start_char", 0) or 0)
        end = int(citation.get("end_char", 0) or 0)
        page_number = int(citation.get("page_number", 0) or 0)
        page_text = pages.get(page_number)
        if page_text is None or end <= start or end > len(page_text):
            verdicts.append({"text": text, "valid_range": False, "reason": "invalid_range"})
            considered += 1
            continue
        window = page_text[start:end]
        considered += 1
        ok = contains_verbatim(text, window) or contains_verbatim(window, text)
        if ok:
            correct += 1
        verdicts.append(
            {
                "text": text,
                "valid_range": ok,
                "window_preview": collapse_whitespace(window)[:160],
            }
        )

    return MetricResult(
        "citation_offset_accuracy",
        (correct / considered) if considered else 0.0,
        "deterministic",
        {"citations": verdicts, "num_citations": considered},
    )
