"""Canonical clause-type vocabulary.

A small local language model asked to name the kind of clause it found does not
answer with one tidy label. It answers with a compound, a near-synonym, or a
phrase borrowed from the contract's own heading:

    "obligation | penalty | termination"   "term_and_termination"
    "Limitation of Liability"              "indemnity"
    "dispute-resolution / arbitration"     "None"

Left alone, those strings make two separate jobs unreliable. The user sees a
scrambled label in the interface, and evaluation scores coverage of the
*expected* clause categories as zero for a clause whose type was in fact
recognized — merely under a different spelling.

This module owns the fix: one canonical vocabulary, applied on the way out of the
model and on the way into the metrics. It splits compound labels, folds aliases
onto their canonical type, and de-duplicates. Anything genuinely unfamiliar is
kept as a normalized label rather than forced into "other", so the vocabulary can
grow without silently discarding information.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

#: The clause categories the project reasons about, and the only labels shown to
#: a user. ``other`` is the explicit fallback for "we read a clause but it does
#: not fit a category we track".
CANONICAL_CLAUSE_TYPES: tuple[str, ...] = (
    "assignment",
    "confidentiality",
    "deadline",
    "dispute_resolution",
    "force_majeure",
    "governing_law",
    "intellectual_property",
    "liability",
    "non_compete",
    "notice",
    "obligation",
    "payment",
    "penalty",
    "renewal",
    "termination",
    "warranty",
    "other",
)

#: Characters that join two clause labels inside one model response.
_SPLIT_RE = re.compile(r"[|/,;+&]|\band\b|\bor\b", re.IGNORECASE)

#: Anything that is not a letter or digit becomes an underscore, so that
#: "Limitation of Liability" and "limitation-of-liability" agree.
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")

#: Aliases folded onto a canonical type. Keys are already slug-normalized.
CLAUSE_TYPE_ALIASES: dict[str, str] = {
    # deadline
    "deadlines": "deadline",
    "duration": "deadline",
    "time_period": "deadline",
    "term": "deadline",
    "contract_term": "deadline",
    "effective_date": "deadline",
    "expiry": "deadline",
    "expiration": "deadline",
    "notice_period": "deadline",
    "date": "deadline",
    # termination
    "terminations": "termination",
    "terminate": "termination",
    "term_and_termination": "termination",
    "terms_and_termination": "termination",
    "termination_for_cause": "termination",
    "termination_for_convenience": "termination",
    "early_termination": "termination",
    "cancellation": "termination",
    "cancellation_rights": "termination",
    "survival": "termination",
    # obligation
    "obligations": "obligation",
    "obligation_of_confidence": "obligation",
    "duty": "obligation",
    "duties": "obligation",
    "responsibility": "obligation",
    "responsibilities": "obligation",
    "covenant": "obligation",
    "performance": "obligation",
    "cooperation": "obligation",
    "restriction": "obligation",
    "restrictions": "obligation",
    # liability
    "liabilities": "liability",
    "limitation_of_liability": "liability",
    "limitation_of_liabilities": "liability",
    "liability_cap": "liability",
    "cap_on_liability": "liability",
    "damages": "liability",
    "indemnity": "liability",
    "indemnification": "liability",
    "indemnities": "liability",
    "indemnify": "liability",
    # confidentiality
    "confidential": "confidentiality",
    "confidential_information": "confidentiality",
    "confidentiality_obligation": "confidentiality",
    "non_disclosure": "confidentiality",
    "nondisclosure": "confidentiality",
    "nda": "confidentiality",
    "secrecy": "confidentiality",
    "trade_secret": "confidentiality",
    "trade_secrets": "confidentiality",
    # dispute_resolution
    "dispute": "dispute_resolution",
    "disputes": "dispute_resolution",
    "arbitration": "dispute_resolution",
    "mediation": "dispute_resolution",
    "adr": "dispute_resolution",
    "jurisdiction": "dispute_resolution",
    "venue": "dispute_resolution",
    "forum_selection": "dispute_resolution",
    # penalty
    "penalties": "penalty",
    "liquidated_damages": "penalty",
    "late_fee": "penalty",
    "late_fees": "penalty",
    "fine": "penalty",
    "fines": "penalty",
    "forfeit": "penalty",
    "forfeiture": "penalty",
    "surcharge": "penalty",
    # payment
    "payments": "payment",
    "fees": "payment",
    "fee": "payment",
    "price": "payment",
    "compensation": "payment",
    "royalty": "payment",
    "invoicing": "payment",
    # governing_law
    "governing_law_and_jurisdiction": "governing_law",
    "applicable_law": "governing_law",
    "choice_of_law": "governing_law",
    # renewal
    "auto_renewal": "renewal",
    "automatic_renewal": "renewal",
    "extension": "renewal",
    # assignment
    "assignment_and_delegation": "assignment",
    "transfer": "assignment",
    "successors": "assignment",
    # non_compete
    "non_competition": "non_compete",
    "noncompete": "non_compete",
    "non_solicitation": "non_compete",
    "exclusivity": "non_compete",
    # warranty
    "warranties": "warranty",
    "representation": "warranty",
    "representations": "warranty",
    "representation_and_warranty": "warranty",
    "guarantee": "warranty",
    # intellectual_property
    "ip": "intellectual_property",
    "intellectual_property_rights": "intellectual_property",
    "ownership": "intellectual_property",
    "license": "intellectual_property",
    "licence": "intellectual_property",
    # force_majeure
    "act_of_god": "force_majeure",
    "impossibility": "force_majeure",
    # notice
    "notices": "notice",
    "notification": "notice",
    "communication": "notice",
    # explicit none
    "none": "other",
    "n_a": "other",
    "na": "other",
    "not_applicable": "other",
    "unknown": "other",
    "unspecified": "other",
    "general": "other",
    "miscellaneous": "other",
    "misc": "other",
}

_CANONICAL_SET = frozenset(CANONICAL_CLAUSE_TYPES)


def slugify_clause_type(label: str) -> str:
    """Reduce one label to a comparison key.

    Args:
        label: A raw label fragment, e.g. ``"Limitation of Liability"``.

    Returns:
        A lower-case, underscore-joined slug, or an empty string when the label
        holds no alphanumeric content.
    """
    return _NON_SLUG_RE.sub("_", label.strip().lower()).strip("_")


def normalize_clause_type(label: str) -> list[str]:
    """Map one raw label onto zero or more canonical clause types.

    A label may name more than one category — ``"obligation | penalty"`` is two
    clauses as far as the taxonomy is concerned, and the caller should emit one
    finding for each so neither is lost.

    Args:
        label: A raw label, possibly compound, from the model.

    Returns:
        Canonical labels in first-seen order, without duplicates. An unrecognized
        but well-formed label is returned as its slug; a blank label yields an
        empty list.
    """
    results: list[str] = []
    for fragment in _SPLIT_RE.split(label or ""):
        slug = slugify_clause_type(fragment)
        if not slug:
            continue
        canonical = CLAUSE_TYPE_ALIASES.get(slug, slug)
        if canonical not in results:
            results.append(canonical)
    return results


def normalize_clause_types(labels: Iterable[str]) -> list[str]:
    """Map many raw labels onto a de-duplicated canonical list.

    Args:
        labels: Raw labels from one or more model responses.

    Returns:
        Canonical labels in first-seen order, without duplicates.
    """
    results: list[str] = []
    for label in labels:
        for canonical in normalize_clause_type(label):
            if canonical not in results:
                results.append(canonical)
    return results


def clause_type_label(label: str) -> str:
    """Render a raw label as the single display string for the user.

    Args:
        label: A raw label, possibly compound.

    Returns:
        Canonical labels joined by ``" / "``, or ``"other"`` when nothing is
        recognized.
    """
    canonical = normalize_clause_type(label)
    return " / ".join(canonical) if canonical else "other"


def is_canonical(label: str) -> bool:
    """Report whether a label is exactly one canonical clause type.

    Args:
        label: Candidate label.

    Returns:
        ``True`` when ``label`` is in :data:`CANONICAL_CLAUSE_TYPES`.
    """
    return label in _CANONICAL_SET


def canonical_clause_vocabulary() -> Sequence[str]:
    """Return the canonical vocabulary in a stable order.

    Returns:
        The tuple of canonical clause types.
    """
    return CANONICAL_CLAUSE_TYPES
