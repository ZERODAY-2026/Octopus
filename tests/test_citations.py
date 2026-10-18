"""Tests for verbatim citation anchoring.

The project promises that every plain-language finding maps back to a
*verifiable quotation* from the document. These tests pin that promise down at
three levels:

* the pure locator (:mod:`app.parsing.citations`) resolves verbatim, repaired and
  unverifiable quotes correctly and returns real character offsets;
* the pipeline stage (:func:`app.pipeline.triage._snap_clause_citations`) replaces
  model text with source text and withholds what it cannot verify;
* the prompt-artifact sanitizer keeps retrieval labels out of the answer.
"""

from __future__ import annotations

import pytest

from app.models import ClauseAnalysis, CitedPassage
from app.parsing.citations import (
    SourceSpan,
    snap_citation,
    snap_citations,
    spans_from_chunks,
)
from app.pipeline.triage import (
    _sanitize_answer_text,
    _snap_clause_citations,
    _spans_from_results,
)

SOURCE = (
    "Either Party may terminate this Agreement for convenience upon thirty (30) "
    "days' prior written notice to the other Party."
)
REFUSAL = "The document is silent on this matter."


# ---------------------------------------------------------------------------
# Pure locator
# ---------------------------------------------------------------------------


def test_verbatim_quote_is_returned_with_exact_offsets() -> None:
    span = SourceSpan(content=SOURCE, page_number=3, start_char=1000)
    match = snap_citation(SOURCE, [span])

    assert match is not None
    assert match.match_type == "verbatim"
    assert match.ratio == 1.0
    assert match.page_number == 3
    # The returned offsets must select exactly the returned text on the page.
    assert SOURCE[match.start_char - 1000 : match.end_char - 1000] == match.text
    assert match.text == SOURCE


def test_whitespace_and_quote_folding_still_counts_as_verbatim() -> None:
    span = SourceSpan(content=SOURCE, page_number=1, start_char=0)
    messy = "Either Party may terminate this Agreement  for convenience upon thirty (30) days\u2019 prior written notice"
    match = snap_citation(messy, [span])

    assert match is not None
    assert match.match_type == "verbatim"
    # The emitted text is the document's characters, not the caller's.
    assert match.text in SOURCE


def test_prefix_quote_is_anchored_to_the_source() -> None:
    span = SourceSpan(content=SOURCE, page_number=1, start_char=0)
    match = snap_citation(
        "Either Party may terminate this Agreement for convenience upon thirty", [span]
    )

    assert match is not None
    assert match.text in SOURCE


def test_altered_quote_is_repaired_to_a_real_source_span() -> None:
    span = SourceSpan(content=SOURCE, page_number=1, start_char=0)
    altered = (
        "Either Party may terminate this Agreement for convenience upon thirty "
        "days prior written notice to the other Party."
    )
    match = snap_citation(altered, [span])

    assert match is not None
    assert match.match_type == "repaired"
    assert 0.0 < match.ratio <= 1.0
    # Even a repaired citation is genuine document text, never model text.
    assert match.text in SOURCE
    assert altered not in SOURCE


def test_paraphrase_is_rejected() -> None:
    span = SourceSpan(content=SOURCE, page_number=1, start_char=0)
    assert snap_citation("The parties can end the contract by giving notice", [span]) is None


def test_refusal_text_is_rejected() -> None:
    span = SourceSpan(content=SOURCE, page_number=1, start_char=0)
    assert snap_citation(REFUSAL, [span]) is None


def test_short_coincidental_overlap_is_rejected() -> None:
    span = SourceSpan(content=SOURCE, page_number=1, start_char=0)
    # "thirty days" is too short to be presented as a quotation.
    assert snap_citation("thirty days", [span]) is None


def test_empty_quote_is_rejected() -> None:
    span = SourceSpan(content=SOURCE, page_number=1, start_char=0)
    assert snap_citation("", [span]) is None
    assert snap_citation("   ", [span]) is None


def test_quote_is_located_in_the_matching_span_and_page() -> None:
    first = SourceSpan(content="A short unrelated clause.", page_number=1, start_char=0)
    second = SourceSpan(content=SOURCE, page_number=7, start_char=500)
    match = snap_citation(SOURCE, [first, second])

    assert match is not None
    assert match.page_number == 7
    assert match.start_char == 500


def test_min_ratio_can_be_tightened() -> None:
    span = SourceSpan(content=SOURCE, page_number=1, start_char=0)
    altered = (
        "Either Party may terminate this Agreement for convenience upon thirty "
        "days prior written notice to the other Party."
    )
    assert snap_citation(altered, [span], min_ratio=0.99) is None


def test_snap_citations_separates_verified_from_rejected() -> None:
    span = SourceSpan(content=SOURCE, page_number=1, start_char=0)
    matches, rejected = snap_citations([SOURCE, REFUSAL, "thirty days"], [span])

    assert len(matches) == 1
    assert matches[0].text == SOURCE
    assert rejected == [REFUSAL, "thirty days"]


# ---------------------------------------------------------------------------
# Span construction and coercion
# ---------------------------------------------------------------------------


def test_spans_from_chunks_accepts_mappings_and_objects() -> None:
    class Chunk:
        content = SOURCE
        page_number = 2
        start_char = 40
        end_char = 40 + len(SOURCE)

    spans = spans_from_chunks(
        [
            {"content": "Some text", "page_number": 1, "start_char": 0, "end_char": 9},
            Chunk(),
            {"content": "   ", "page_number": 3},
        ]
    )

    assert len(spans) == 2
    assert spans[1].page_number == 2
    assert spans[1].start_char == 40


def test_source_span_derives_end_char_when_absent() -> None:
    span = SourceSpan(content="abcde", start_char=10)
    assert span.end_char == 15


def test_spans_from_results_reads_nested_metadata() -> None:
    results = [
        {
            "content": SOURCE,
            "metadata": {"page_number": "2", "start_char": "100", "end_char": "221"},
        },
        {"content": "", "metadata": {}},
        {"content": "another clause", "metadata": {}},
    ]
    spans = _spans_from_results(results)

    assert len(spans) == 2
    assert spans[0].page_number == 2
    assert spans[0].start_char == 100
    assert spans[0].end_char == 221
    assert spans[1].page_number == 1


# ---------------------------------------------------------------------------
# Pipeline stage: anchoring and withholding
# ---------------------------------------------------------------------------


def _clause(*passages: tuple[str, int, int, int]) -> ClauseAnalysis:
    return ClauseAnalysis(
        clause_type="termination",
        plain_language_summary="Termination for convenience requires notice.",
        cited_passages=[
            CitedPassage(text=text, page_number=page, start_char=start, end_char=end)
            for text, page, start, end in passages
        ],
        confidence="high",
    )


def test_snap_clause_citations_replaces_text_and_offsets() -> None:
    spans = [SourceSpan(content=SOURCE, page_number=1, start_char=0)]
    clauses, rejected, dropped = _snap_clause_citations(
        [_clause((SOURCE, 1, 0, len(SOURCE)))], spans
    )

    assert rejected == []
    assert dropped == 0
    passage = clauses[0].cited_passages[0]
    assert passage.text == SOURCE
    assert passage.start_char == 0
    assert passage.end_char == len(SOURCE)


def test_uncited_clause_is_withheld() -> None:
    spans = [SourceSpan(content=SOURCE, page_number=1, start_char=0)]
    clauses, rejected, dropped = _snap_clause_citations([_clause((REFUSAL, 1, 0, 20))], spans)

    # A clause whose only quotation cannot be verified is not a finding.
    assert clauses == []
    assert dropped == 1
    assert rejected == [REFUSAL]


def test_verifiable_clause_survives_an_unverifiable_sibling() -> None:
    spans = [SourceSpan(content=SOURCE, page_number=1, start_char=0)]
    clauses, rejected, dropped = _snap_clause_citations(
        [_clause((SOURCE, 1, 0, 10), (REFUSAL, 1, 0, 30))], spans
    )

    assert dropped == 0
    assert rejected == [REFUSAL]
    assert len(clauses[0].cited_passages) == 1
    assert clauses[0].cited_passages[0].text == SOURCE


# ---------------------------------------------------------------------------
# Prompt-artifact sanitizer
# ---------------------------------------------------------------------------


def test_excerpt_markers_are_stripped_from_answers() -> None:
    raw = f"{SOURCE} [EXCERPT 2 | Page: 1 | Char offset: 5143-5725]"
    assert _sanitize_answer_text(raw) == SOURCE


def test_self_citation_sentences_are_removed() -> None:
    raw = f"{SOURCE} This is stated in excerpts 3.3 and 2.3."
    assert _sanitize_answer_text(raw) == SOURCE


def test_section_self_citation_is_removed_without_eating_digits() -> None:
    raw = "This is stated in Section 3.1. The notice period is thirty (30) days."
    assert _sanitize_answer_text(raw) == "The notice period is thirty (30) days."


@pytest.mark.parametrize(
    "text",
    [
        "",
        "The document is silent on this matter.",
        "Either Party may terminate this Agreement.",
    ],
)
def test_sanitizer_leaves_normal_text_untouched(text: str) -> None:
    assert _sanitize_answer_text(text) == text
