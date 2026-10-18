"""Unit tests for the RAGAS-style metric implementations.

These tests exercise the metric maths directly, with no pipeline and no model,
so a regression in a formula fails here rather than silently shifting an
aggregate score. Both the deterministic retrieval metrics and the lexical
approximations of the judged metrics are covered, plus the local LLM-judge code
path via a scripted double.
"""

from __future__ import annotations

import pytest

from app.evaluation import metrics
from app.evaluation.metrics import (
    LLMJudge,
    MetricResult,
    answer_correctness,
    answer_relevancy,
    citation_grounding_rate,
    citation_locator_accuracy,
    citation_offset_accuracy,
    context_precision,
    context_recall,
    context_relevance,
    faithfulness,
    hit_rate_at_k,
    is_abstention,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.text import (
    contains_verbatim,
    coverage,
    extract_numbers,
    longest_common_span_ratio,
    sentences,
    tokenize,
)
from tests.support import RelevanceJudgeLLM, run_async

CLAUSE = (
    "Either Party may terminate this Agreement for convenience upon thirty (30) "
    "days' prior written notice to the other Party."
)


# ---------------------------------------------------------------------------
# Deterministic retrieval metrics
# ---------------------------------------------------------------------------


def test_precision_at_k() -> None:
    assert precision_at_k(["a", "b", "c"], {"a"}, 2) == pytest.approx(0.5)
    assert precision_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    assert precision_at_k(["a", "b"], {"z"}, 2) == pytest.approx(0.0)
    assert precision_at_k(["a"], {"a"}, 0) == 0.0
    assert precision_at_k([], {"a"}, 3) == 0.0


def test_recall_at_k() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == pytest.approx(1.0)
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, 1) == pytest.approx(0.5)
    # No ground-truth items to find is a vacuous success, not a failure.
    assert recall_at_k(["a"], set(), 1) == 1.0


def test_hit_rate_at_k() -> None:
    assert hit_rate_at_k(["x", "a"], {"a"}, 2) == 1.0
    assert hit_rate_at_k(["x", "y"], {"a"}, 2) == 0.0
    assert hit_rate_at_k(["a"], {"a"}, 0) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b"], {"a"}) == pytest.approx(1.0)
    assert reciprocal_rank(["x", "a"], {"a"}) == pytest.approx(0.5)
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_rewards_early_hits() -> None:
    perfect = ndcg_at_k(["a", "b"], {"a", "b"}, 2)
    reversed_order = ndcg_at_k(["b", "a"], {"a", "b"}, 2)
    assert perfect == pytest.approx(1.0)
    assert reversed_order == pytest.approx(1.0)  # both hits, order irrelevant at k=2
    assert ndcg_at_k(["x", "a"], {"a"}, 2) < perfect


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0
    assert ndcg_at_k([], {"a"}, 2) == 0.0


# ---------------------------------------------------------------------------
# Text utilities the metrics depend on
# ---------------------------------------------------------------------------


def test_tokenize_drops_stopwords_and_stems() -> None:
    tokens = tokenize("The parties shall terminate the agreement upon notice")
    assert "the" not in tokens
    assert "terminat" in tokens  # "terminate" stemmed
    assert "notice" in tokens


def test_extract_numbers_reads_digits_and_words() -> None:
    numbers = extract_numbers("within thirty (30) days and fifty percent (50%)")
    assert {"30", "50"} <= numbers
    assert extract_numbers(CLAUSE) == {"30"}


def test_contains_verbatim_tolerates_pdf_whitespace() -> None:
    assert contains_verbatim("may terminate this Agreement", CLAUSE)
    assert not contains_verbatim("may terminate the agreement", CLAUSE)


def test_longest_common_span_ratio_scales_with_overlap() -> None:
    assert longest_common_span_ratio(CLAUSE, CLAUSE) == pytest.approx(1.0)
    partial = "Either Party may terminate this Agreement and also fly to the moon"
    assert 0.3 < longest_common_span_ratio(partial, CLAUSE) < 1.0
    assert longest_common_span_ratio("completely unrelated words here", CLAUSE) == 0.0


def test_coverage_measures_reference_terms_present() -> None:
    assert coverage("terminate convenience notice", CLAUSE, n=1) == pytest.approx(1.0)
    assert coverage("terminate quantum", CLAUSE, n=1) == pytest.approx(0.5)


def test_sentences_splits_on_terminal_punctuation() -> None:
    text = "First sentence. Second sentence! Third?"
    assert sentences(text) == ["First sentence.", "Second sentence!", "Third?"]


# ---------------------------------------------------------------------------
# Faithfulness / groundedness
# ---------------------------------------------------------------------------


def test_faithfulness_grounded_answer_scores_one() -> None:
    result = run_async(faithfulness(CLAUSE, [CLAUSE]))
    assert result.score == pytest.approx(1.0)
    assert result.details["unsupported_claims"] == []


def test_faithfulness_flags_a_hallucinated_number() -> None:
    answer = "Either Party may terminate for convenience upon ninety (90) days' notice."
    result = run_async(faithfulness(answer, [CLAUSE]))
    assert result.score < 1.0
    assert result.details["unsupported_claims"]


def test_faithfulness_empty_answer_is_zero() -> None:
    result = run_async(faithfulness("", [CLAUSE]))
    assert result.score == 0.0
    assert result.details["reason"] == "empty_answer"


def test_faithfulness_without_context_is_zero() -> None:
    result = run_async(faithfulness(CLAUSE, []))
    assert result.score == 0.0
    assert result.details["reason"] == "no_context_provided"


def test_faithfulness_of_abstention_is_one() -> None:
    result = run_async(faithfulness("The document is silent on this matter.", [CLAUSE]))
    assert result.score == pytest.approx(1.0)
    assert result.details["abstention"] is True


def test_faithfulness_uses_judge_when_supplied() -> None:
    judge = LLMJudge(RelevanceJudgeLLM(support_threshold=0.6))
    result = run_async(faithfulness(CLAUSE, [CLAUSE], judge))
    assert result.mode == "llm"
    assert result.details["total_claims"] >= 1


# ---------------------------------------------------------------------------
# Context relevance, precision, recall
# ---------------------------------------------------------------------------


def test_context_relevance_counts_on_topic_chunks() -> None:
    contexts = [CLAUSE, "The governing law of this agreement is Delaware."]
    result = run_async(context_relevance("What notice is required to terminate?", contexts))
    assert 0.0 < result.score < 1.0
    assert result.details["num_chunks"] == 2


def test_context_relevance_without_context_is_zero() -> None:
    result = run_async(context_relevance("anything", []))
    assert result.score == 0.0


def test_context_precision_rewards_early_relevance() -> None:
    on_topic = CLAUSE
    off_topic = "The parties are located in different countries."
    first = run_async(context_precision("terminate convenience notice", [on_topic, off_topic]))
    second = run_async(context_precision("terminate convenience notice", [off_topic, on_topic]))
    assert first.score > second.score


def test_context_recall_measures_reference_support() -> None:
    result = run_async(context_recall(CLAUSE, [CLAUSE]))
    assert result.score == pytest.approx(1.0)
    missing = run_async(context_recall("The agreement contains an arbitration clause.", [CLAUSE]))
    assert missing.score < 1.0


# ---------------------------------------------------------------------------
# Answer correctness and relevancy
# ---------------------------------------------------------------------------


def test_answer_correctness_of_identical_text_is_one() -> None:
    result = run_async(answer_correctness(CLAUSE, CLAUSE))
    assert result.score == pytest.approx(1.0)
    assert result.details["f1"] == pytest.approx(1.0)


def test_answer_correctness_of_unrelated_text_is_low() -> None:
    result = run_async(answer_correctness("Bananas are yellow fruit.", CLAUSE))
    assert result.score < 0.2


def test_answer_relevancy_prefers_the_on_topic_answer() -> None:
    on_topic = run_async(
        answer_relevancy("What notice is required to terminate?", CLAUSE)
    )
    off_topic = run_async(
        answer_relevancy("What notice is required to terminate?", "The fee is $12,000.")
    )
    assert on_topic.score > off_topic.score


def test_answer_relevancy_caps_an_abstention() -> None:
    result = run_async(
        answer_relevancy(
            "What notice is required to terminate?",
            "The document is silent on this matter and no conclusion can be drawn.",
        )
    )
    assert result.score <= 0.35


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------


def test_citation_grounding_rate_accepts_verbatim_and_rejects_fabrication() -> None:
    result = citation_grounding_rate([CLAUSE, "A totally invented clause."], [CLAUSE])
    assert result.score == pytest.approx(0.5)
    verdicts = {v["quote"]: v for v in result.details["citations"]}
    assert verdicts[CLAUSE]["grounded"] is True
    assert verdicts["A totally invented clause."]["grounded"] is False


def test_citation_grounding_rate_with_no_citations_is_vacuously_one() -> None:
    assert citation_grounding_rate([], [CLAUSE]).score == 1.0
    assert citation_grounding_rate([CLAUSE], []).score == 0.0


def test_citation_locator_accuracy_detects_wrong_page() -> None:
    pages = [
        {"page_number": 1, "text": "Unrelated page one."},
        {"page_number": 2, "text": CLAUSE},
    ]
    right = citation_locator_accuracy(
        [{"text": CLAUSE, "page_number": 2}], pages
    )
    wrong = citation_locator_accuracy(
        [{"text": CLAUSE, "page_number": 1}], pages
    )
    assert right.score == pytest.approx(1.0)
    assert wrong.score == pytest.approx(0.0)
    assert wrong.details["citations"][0]["reason"] == "wrong_page"


def test_citation_offset_accuracy_validates_the_character_window() -> None:
    page_text = "0000" + CLAUSE + "9999"
    pages = [{"page_number": 1, "text": page_text}]
    good = citation_offset_accuracy(
        [{"text": CLAUSE, "page_number": 1, "start_char": 4, "end_char": 4 + len(CLAUSE)}],
        pages,
    )
    bad = citation_offset_accuracy(
        [{"text": CLAUSE, "page_number": 1, "start_char": 0, "end_char": 3}],
        pages,
    )
    assert good.score == pytest.approx(1.0)
    assert bad.score == pytest.approx(0.0)


def test_citation_offset_accuracy_rejects_out_of_range() -> None:
    pages = [{"page_number": 1, "text": "short"}]
    result = citation_offset_accuracy(
        [{"text": "short", "page_number": 1, "start_char": 0, "end_char": 999}], pages
    )
    assert result.score == 0.0
    assert result.details["citations"][0]["reason"] == "invalid_range"


# ---------------------------------------------------------------------------
# Abstention detection and result container
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The document is silent on this matter.",
        "No provision addresses the requested subject.",
        "The text does not specify a venue.",
        "This cannot be determined from the excerpts.",
    ],
)
def test_is_abstention_recognizes_calibrated_refusals(text: str) -> None:
    assert is_abstention(text) is True


def test_is_abstention_is_false_for_a_committal_answer() -> None:
    assert is_abstention("The notice period is thirty (30) days.") is False


def test_is_abstention_is_false_for_partial_answer_with_silence_tail() -> None:
    text = (
        "remains in effect until terminated by either Party upon thirty (30) days' "
        "prior written notice. The excerpts do not specify a fixed termination date."
    )
    assert is_abstention(text) is False


def test_metric_result_serializes_with_rounded_score() -> None:
    payload = MetricResult("demo", 0.1234567, "lexical", {"k": 1}).to_dict()
    assert payload["score"] == 0.123457
    assert payload["mode"] == "lexical"
    assert payload["details"] == {"k": 1}


def test_llm_judge_paths_use_the_scripted_double() -> None:
    judge = LLMJudge(RelevanceJudgeLLM())
    claims = run_async(judge.extract_claims(CLAUSE))
    assert claims, "the double returns claims"
    assert run_async(judge.verify_claim(CLAUSE, CLAUSE)) is True
    questions = run_async(judge.generate_questions(CLAUSE))
    assert questions


def test_metrics_module_exposes_lexical_constants_for_auditing() -> None:
    # Thresholds are tuning inputs, not magic numbers: keep them documented.
    assert 0.0 < metrics.LEXICAL_SUPPORT_THRESHOLD < 1.0
    assert metrics.ABSTENTION_MARKERS
