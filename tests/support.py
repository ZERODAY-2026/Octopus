"""Shared test infrastructure for the Contracter test suite.

This module holds the pieces that more than one test file needs:

* a socket-level **network blocker** that makes outbound traffic impossible,
  used to prove the off-line guarantee rather than assert it;
* **LLM test doubles** — a scripted client, a client that always fails, and two
  behaviourally realistic analysis clients: one that answers strictly from the
  retrieved excerpts, and one that tampers with numbers and fabricates a clause.
  The pair is what proves the evaluation metrics can actually detect a bad
  system instead of rubber-stamping a good one;
* helpers for building a hermetic :class:`~app.pipeline.triage.TriagePipeline`.
"""

from __future__ import annotations

import json
import re
import socket
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from app.config import Settings
from app.embeddings.deterministic import OfflineHashingEmbedder
from app.evaluation.text import coverage, normalize_for_match
from app.llm.inference import LLMClient
from app.pipeline.triage import TriagePipeline
from app.vectorstore.store import VectorStore

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_documents"

# ---------------------------------------------------------------------------
# Network blocking
# ---------------------------------------------------------------------------


class NetworkBlockedError(RuntimeError):
    """Raised when code under test attempts to touch the network."""


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "", "::"}


def _is_loopback(address: Any) -> bool:
    """Decide whether a socket address targets the local machine only.

    Args:
        address: The address passed to ``socket.connect``, or a bare hostname
            (``getaddrinfo``/``gethostbyname`` supply a host string, sometimes
            as bytes).

    Returns:
        True for loopback and UNIX-domain targets.
    """
    if isinstance(address, bytes):
        address = address.decode("utf-8", errors="replace")
    if isinstance(address, str):  # bare hostname from the resolver
        return address in _LOOPBACK_HOSTS
    if not isinstance(address, (tuple, list)) or not address:
        return False
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode("utf-8", errors="replace")
    host = str(host)
    if host in _LOOPBACK_HOSTS:
        return True
    return host.startswith("127.") or host.startswith("::1")


@contextmanager
def block_network(
    *,
    allow_loopback: bool = True,
    attempts: list[Any] | None = None,
) -> Iterator[list[Any]]:
    """Make outbound network access impossible for the duration of the block.

    Patches the socket layer so that every connect, address lookup and
    higher-level convenience call is inspected. Loopback traffic can be allowed
    because a strictly local model server is not a privacy leak, but DNS for any
    external hostname is disabled outright.

    Args:
        allow_loopback: Permit connections to 127.0.0.0/8 and ::1.
        attempts: Optional list that receives every blocked address, so a test
            can assert that nothing was even attempted.

    Yields:
        The list recording blocked connection attempts.
    """
    recorded: list[Any] = attempts if attempts is not None else []

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname

    def _guard(address: Any, action: str) -> None:
        if allow_loopback and _is_loopback(address):
            return
        recorded.append({"address": address, "action": action})
        raise NetworkBlockedError(
            f"Network access blocked by the Contracter privacy test harness: "
            f"{action} to {address!r}"
        )

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        _guard(address, "connect")
        return real_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
        _guard(address, "connect_ex")
        return real_connect_ex(self, address)

    def guarded_create_connection(
        address: Any, *args: Any, **kwargs: Any
    ) -> Any:
        _guard(address, "create_connection")
        return real_create_connection(address, *args, **kwargs)

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not (allow_loopback and str(host) in _LOOPBACK_HOSTS):
            recorded.append({"address": host, "action": "getaddrinfo"})
            raise NetworkBlockedError(
                f"DNS resolution blocked by the Contracter privacy test harness: {host!r}"
            )
        return real_getaddrinfo(host, *args, **kwargs)

    def guarded_gethostbyname(host: Any) -> Any:
        if not (allow_loopback and str(host) in _LOOPBACK_HOSTS):
            recorded.append({"address": host, "action": "gethostbyname"})
            raise NetworkBlockedError(
                f"DNS resolution blocked by the Contracter privacy test harness: {host!r}"
            )
        return real_gethostbyname(host)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
    socket.gethostbyname = guarded_gethostbyname  # type: ignore[assignment]

    try:
        yield recorded
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = real_connect_ex  # type: ignore[method-assign]
        socket.create_connection = real_create_connection  # type: ignore[assignment]
        socket.getaddrinfo = real_getaddrinfo  # type: ignore[assignment]
        socket.gethostbyname = real_gethostbyname  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# LLM test doubles
# ---------------------------------------------------------------------------


class ScriptedLLMClient(LLMClient):
    """LLM client that returns queued or callback-produced text without any I/O."""

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        responder: Callable[[str, str | None], str] | None = None,
    ) -> None:
        """Initialize the double.

        Args:
            responses: Fixed responses returned in order; the last one repeats.
            responder: Callback taking ``(prompt, system_prompt)`` and returning
                a response. Takes precedence over ``responses``.
        """
        # Deliberately do not call super().__init__ with a real base URL: the
        # double must never be able to open a connection by accident.
        self.base_url = "http://127.0.0.1:0"
        self.model_name = "scripted-test-model"
        self.timeout = 1.0
        self._client = None
        self._responses = list(responses or [])
        self._responder = responder
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> str:
        """Record the call and return the scripted response.

        Args:
            prompt: Prompt body.
            system_prompt: System instructions.
            temperature: Sampling temperature (ignored).

        Returns:
            Scripted response text.
        """
        self.calls.append(
            {"prompt": prompt, "system_prompt": system_prompt, "temperature": temperature}
        )
        if self._responder is not None:
            return self._responder(prompt, system_prompt)
        if not self._responses:
            return ""
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)

    async def health_check(self) -> bool:
        """Report a healthy model without contacting anything.

        Returns:
            Always True.
        """
        return True

    async def close(self) -> None:
        """Mark the double as closed."""
        self.closed = True


class FailingLLMClient(ScriptedLLMClient):
    """LLM client that always raises, to exercise error handling paths."""

    def __init__(self, message: str = "connection refused (simulated)") -> None:
        """Initialize the double.

        Args:
            message: Error text raised on every call.
        """
        super().__init__(responder=lambda *_: "")
        self.message = message

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> str:
        """Raise the configured error.

        Args:
            prompt: Ignored.
            system_prompt: Ignored.
            temperature: Ignored.

        Raises:
            RuntimeError: Always.
        """
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, "temperature": temperature})
        raise RuntimeError(self.message)


class CapturingLLMClient(ScriptedLLMClient):
    """Records every request/response pair while delegating to another client."""

    def __init__(self, delegate: LLMClient | None = None) -> None:
        """Initialize the wrapper.

        Args:
            delegate: Client that produces the response.
        """
        super().__init__(responder=lambda *_: "")
        self.delegate = delegate

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> str:
        """Capture the call, then delegate.

        Args:
            prompt: Prompt body.
            system_prompt: System instructions.
            temperature: Sampling temperature.

        Returns:
            The delegate's response.
        """
        self.calls.append(
            {"prompt": prompt, "system_prompt": system_prompt, "temperature": temperature}
        )
        if self.delegate is None:
            return ""
        return await self.delegate.generate(prompt, system_prompt, temperature)


# ---------------------------------------------------------------------------
# Excerpt parsing for behavioural LLM doubles
# ---------------------------------------------------------------------------

_EXCERPT_RE = re.compile(
    r"\[EXCERPT (\d+) \| Page: (\d+) \| Char offset: (\d+)-(\d+)\]\s*\n(.*?)"
    r"(?=\n\n\[EXCERPT |\n-{10,}|\Z)",
    re.DOTALL,
)
_QUESTION_RE = re.compile(r"USER INQUIRY:\s*\n(.*?)\n", re.DOTALL)

#: Clause-type heuristics used by the behavioural doubles. Every matching
#: category is emitted, because a real clause genuinely is multi-label: a
#: "thirty (30) days' written notice to terminate" provision is both a
#: ``termination`` and a ``deadline``. Ordered most-specific first so the cap
#: keeps the informative labels.
_CLAUSE_TYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("confidentiality", ("confidential", "disclosing party", "receiving party")),
    ("penalty", ("interest at", "per month", "penalt", "liquidated", "accrue", "unpaid")),
    ("liability", ("liability", "indemnif", "hold harmless", "infringe")),
    ("dispute_resolution", ("governed by", "governing law", "arbitration", "jurisdiction", "courts")),
    ("termination", ("terminat", "non-renewal", "renewing", "survive", "cure")),
    ("deadline", ("days", "notice", "milestone", "period of")),
    ("obligation", ("shall", "must", "will deliver", "responsible for")),
)


def parse_excerpts(prompt: str) -> list[dict[str, Any]]:
    """Extract the excerpt blocks that the pipeline injected into a prompt.

    Args:
        prompt: The rendered query or summary prompt.

    Returns:
        List of dicts with ``text``, ``page_number``, ``start_char`` and ``end_char``.
    """
    excerpts: list[dict[str, Any]] = []
    for match in _EXCERPT_RE.finditer(prompt):
        excerpts.append(
            {
                "index": int(match.group(1)),
                "page_number": int(match.group(2)),
                "start_char": int(match.group(3)),
                "end_char": int(match.group(4)),
                "text": match.group(5).strip(),
            }
        )
    return excerpts


def parse_question(prompt: str) -> str:
    """Extract the user inquiry from a rendered query prompt.

    Args:
        prompt: The rendered query prompt.

    Returns:
        The question text, or an empty string when absent.
    """
    match = _QUESTION_RE.search(prompt)
    return match.group(1).strip() if match else ""


def classify_clauses(text: str, max_types: int = 2) -> list[str]:
    """Assign every applicable clause category to an excerpt.

    Args:
        text: Excerpt text.
        max_types: Maximum categories to return.

    Returns:
        List of clause categories, most specific first. Falls back to ``["other"]``.
    """
    lowered = text.lower()
    matched = [
        clause_type
        for clause_type, needles in _CLAUSE_TYPE_RULES
        if any(needle in lowered for needle in needles)
    ]
    if not matched:
        return ["other"]
    return matched[:max_types]


def first_sentence(text: str, max_chars: int = 320) -> str:
    """Return the first sentence of a passage, trimmed to a length budget.

    Args:
        text: Source passage.
        max_chars: Maximum characters to return.

    Returns:
        A verbatim prefix of the passage.
    """
    collapsed = " ".join(text.split())
    if not collapsed:
        return ""
    match = re.search(r".+?[.;](?=\s|$)", collapsed)
    sentence = match.group(0).strip() if match else collapsed
    if len(sentence) > max_chars:
        sentence = sentence[:max_chars].rstrip()
    return sentence


class GroundedAnalysisLLM(ScriptedLLMClient):
    """Simulates a well-behaved model that is strictly faithful to its excerpts.

    Retrieval is still allowed to fail: when none of the retrieved excerpts are
    on-topic for the question, this double abstains and says the document is
    silent, which is exactly the calibrated-uncertainty behaviour the plan
    requires. That makes it a realistic stand-in for a good pipeline and lets
    the abstention metrics be tested end to end.
    """

    def __init__(self, *, relevance_threshold: float = 0.25, max_excerpts: int = 4) -> None:
        """Initialize the double.

        Args:
            relevance_threshold: Question-term coverage an excerpt must reach
                before it is treated as on-topic. Kept low because a competent
                model still uses a partially-overlapping excerpt; the threshold
                only guards against entirely off-topic evidence.
            max_excerpts: Maximum excerpts to draw clauses from.
        """
        super().__init__(responder=self._respond)
        self.relevance_threshold = relevance_threshold
        self.max_excerpts = max_excerpts

    def _respond(self, prompt: str, system_prompt: str | None) -> str:
        """Produce a grounded analysis payload for the given prompt.

        Args:
            prompt: Rendered pipeline prompt.
            system_prompt: System instructions (unused).

        Returns:
            JSON string matching the pipeline's expected schema.
        """
        excerpts = parse_excerpts(prompt)
        question = parse_question(prompt)

        if not excerpts:
            return json.dumps(self._abstention_payload("no retrieval results"))

        if question:
            relevant = [
                excerpt
                for excerpt in excerpts
                if coverage(question, excerpt["text"], n=1) >= self.relevance_threshold
            ]
        else:
            relevant = excerpts

        if not relevant:
            return json.dumps(self._abstention_payload("no on-topic excerpt"))

        clauses: list[dict[str, Any]] = []
        summaries: list[str] = []
        for excerpt in relevant[: self.max_excerpts]:
            quotation = first_sentence(excerpt["text"])
            if not quotation:
                continue
            summary = f"The document states that {quotation[0].lower() + quotation[1:]}"
            for clause_type in classify_clauses(excerpt["text"]):
                clauses.append(
                    {
                        "clause_type": clause_type,
                        "plain_language_summary": summary,
                        "cited_passages": [
                            {
                                "text": quotation,
                                "page_number": excerpt["page_number"],
                                "start_char": excerpt["start_char"],
                                "end_char": excerpt["end_char"],
                            }
                        ],
                        "confidence": "high",
                    }
                )
            summaries.append(summary)

        answer = " ".join(summaries) if summaries else "No clauses were identified in the excerpts."
        return json.dumps(
            {
                "answer": answer,
                "answer_status": "answered" if summaries else "silent",
                "clauses": clauses,
                "uncertainty_notes": [
                    "Notice delivery methods and any forum selection not stated above are not addressed by the document.",
                    "Review by a licensed attorney is recommended for negotiated or high-value terms.",
                ],
            }
        )

    def _abstention_payload(self, reason: str) -> dict[str, Any]:
        """Build the calibrated-refusal payload used by the behavioural doubles.

        Args:
            reason: Diagnostic reason recorded in the payload.

        Returns:
            A refusal payload matching the pipeline's schema.
        """
        return {
            "answer": (
                "The retrieved excerpts are silent on this question; the document does "
                "not address it, and no conclusion can be drawn."
            ),
            "answer_status": "silent",
            "clauses": [],
            "uncertainty_notes": [
                "The document is silent on this matter.",
                "No provision addressing the requested subject was found in the document.",
            ],
            "reason": reason,
        }


class HallucinatingAnalysisLLM(GroundedAnalysisLLM):
    """Simulates a plausible but unfaithful model.

    Numbers are shifted and an unsupported penalty clause with a fabricated
    citation is appended. The generated text stays fluent and contract-like, so
    the failure is only detectable by a real groundedness check — which is the
    point of having one.
    """

    #: Corruption applied to numeric terms so the answer contradicts the source.
    NUMBER_SWAPS: tuple[tuple[str, str], ...] = (
        ("thirty", "ninety"),
        ("sixty", "ninety"),
        ("fifteen", "forty-five"),
        ("thirty (30)", "ninety (90)"),
        ("sixty (60)", "ninety (90)"),
        ("two (2) years", "ten (10) years"),
        ("five (5) business days", "twenty (20) business days"),
        ("ninety (90)", "one hundred twenty (120)"),
    )

    #: A quotation that appears in none of the sample documents.
    FABRICATED_QUOTE: str = (
        "The Receiving Party shall pay a penalty of two hundred fifty thousand "
        "dollars ($250,000) for each unauthorised disclosure."
    )

    def _respond(self, prompt: str, system_prompt: str | None) -> str:
        """Produce an unfaithful analysis payload.

        Args:
            prompt: Rendered pipeline prompt.
            system_prompt: System instructions (unused).

        Returns:
            JSON string containing shifted numbers and a fabricated citation.
        """
        payload = json.loads(super()._respond(prompt, system_prompt))

        def corrupt(text: str) -> str:
            for original, replacement in self.NUMBER_SWAPS:
                text = text.replace(original, replacement)
            return text

        payload["answer"] = corrupt(payload.get("answer", ""))
        for clause in payload.get("clauses", []):
            clause["plain_language_summary"] = corrupt(clause.get("plain_language_summary", ""))
            for passage in clause.get("cited_passages", []):
                passage["text"] = corrupt(passage.get("text", ""))

        payload["clauses"].append(
            {
                "clause_type": "penalty",
                "plain_language_summary": (
                    "The document imposes a liquidated penalty of $250,000 for each "
                    "unauthorised disclosure of confidential information."
                ),
                "cited_passages": [
                    {
                        "text": self.FABRICATED_QUOTE,
                        "page_number": 1,
                        "start_char": 0,
                        "end_char": len(self.FABRICATED_QUOTE),
                    }
                ],
                "confidence": "high",
            }
        )
        payload["answer"] = (
            payload.get("answer", "")
            + " The document imposes a liquidated penalty of $250,000 per unauthorised disclosure."
        ).strip()
        return json.dumps(payload)


class AbstainingAnalysisLLM(ScriptedLLMClient):
    """A perfectly calibrated model: it always declines when evidence is absent.

    Unanswerability cannot be detected by a keyword heuristic, so the
    calibrated-uncertainty metrics are proven against purpose-built doubles
    rather than a heuristic one. This double models the ideal behaviour: it
    answers only when the caller marks the question as answerable, and abstains
    otherwise.
    """

    def __init__(self, answerable_ids: Sequence[str] | None = None) -> None:
        """Initialize the double.

        Args:
            answerable_ids: Question texts this double treats as answerable. Any
                other question receives a calibrated refusal.
        """
        super().__init__(responder=self._respond)
        self.answerable_ids = list(answerable_ids or [])

    def _respond(self, prompt: str, system_prompt: str | None) -> str:
        """Abstain unless the question is on the allow-list.

        Args:
            prompt: Rendered pipeline prompt.
            system_prompt: System instructions (unused).

        Returns:
            JSON payload, either an answer or a calibrated refusal.
        """
        question = parse_question(prompt)
        if question and question not in self.answerable_ids:
            return json.dumps(
                {
                    "answer": (
                        "The document is silent on this point: no provision in the retrieved "
                        "excerpts addresses it, so no conclusion is offered."
                    ),
                    "answer_status": "silent",
                    "clauses": [],
                    "uncertainty_notes": ["The document is silent on this matter."],
                }
            )
        return GroundedAnalysisLLM()._respond(prompt, system_prompt)


class ConfabulatingAnalysisLLM(ScriptedLLMClient):
    """An over-answering model that never admits the document is silent.

    Used to prove the abstention metrics fail a system that fabricates answers
    to unanswerable questions, which is the single most dangerous failure mode
    for a legal triage tool.
    """

    def __init__(self) -> None:
        """Register the confabulating responder (was previously never wired)."""
        super().__init__(responder=self._respond)

    def _respond(self, prompt: str, system_prompt: str | None) -> str:
        """Always answer, inventing a provision when the excerpts are unhelpful.

        Args:
            prompt: Rendered pipeline prompt.
            system_prompt: System instructions (unused).

        Returns:
            JSON payload that never contains a refusal.
        """
        payload = json.loads(GroundedAnalysisLLM()._respond(prompt, system_prompt))
        if not payload.get("clauses"):
            payload["clauses"] = [
                {
                    "clause_type": "penalty",
                    "plain_language_summary": (
                        "The agreement imposes a fifty thousand dollar ($50,000) fee for the "
                        "matter raised."
                    ),
                    "cited_passages": [
                        {
                            "text": (
                                "A fee of fifty thousand dollars ($50,000) shall be payable in "
                                "respect of the matter raised."
                            ),
                            "page_number": 1,
                            "start_char": 0,
                            "end_char": 95,
                        }
                    ],
                    "confidence": "high",
                }
            ]
            payload["answer"] = (
                "The agreement imposes a fifty thousand dollar ($50,000) fee for the matter raised."
            )
        payload["uncertainty_notes"] = []
        payload["answer_status"] = "answered"
        return json.dumps(payload)


class RelevanceJudgeLLM(ScriptedLLMClient):
    """Judge double used to exercise the LLM-judged metric path deterministically.

    It answers claim-verification prompts by checking content-token coverage of
    the evidence, and question-generation prompts with a fixed list. That keeps
    the judged code path fully exercised without any model or network.
    """

    def __init__(self, *, support_threshold: float = 0.6, questions: Sequence[str] | None = None) -> None:
        """Initialize the judge double.

        Args:
            support_threshold: Coverage above which a claim is supported.
            questions: Questions returned by the reverse-question step.
        """
        super().__init__(responder=self._respond)
        self.support_threshold = support_threshold
        self.questions = list(
            questions
            or [
                "What are the termination conditions in the agreement?",
                "What notice period applies to termination of the agreement?",
                "What are the termination requirements?",
            ]
        )

    def _respond(self, prompt: str, system_prompt: str | None) -> str:
        """Answer a judge prompt.

        Args:
            prompt: Judge prompt.
            system_prompt: Judge role instructions, used to pick the task.

        Returns:
            A one-word verdict or a JSON list, depending on the prompt type.
        """
        system = system_prompt or ""
        if "write the questions" in system:
            return json.dumps({"questions": self.questions})
        if "decompose text into atomic" in system:
            # Two claims per call is enough to exercise the aggregation maths.
            return json.dumps({"claims": ["The agreement may be terminated for convenience.", "Written notice is required."]})

        evidence_match = re.search(r"EVIDENCE:\s*\"\"\"(.*?)\"\"\"", prompt, re.DOTALL)
        claim_match = re.search(r"CLAIM:\s*\"\"\"(.*?)\"\"\"", prompt, re.DOTALL)
        if evidence_match and claim_match:
            evidence = evidence_match.group(1)
            claim = claim_match.group(1)
            return "yes" if coverage(claim, evidence, n=1) >= self.support_threshold else "no"

        context_match = re.search(r"RETRIEVED EXCERPT:\s*\"\"\"(.*?)\"\"\"", prompt, re.DOTALL)
        question_match = re.search(r"QUESTION:\s*\"\"\"(.*?)\"\"\"", prompt, re.DOTALL)
        if context_match and question_match:
            return (
                "yes"
                if coverage(question_match.group(1), context_match.group(1), n=1)
                >= self.support_threshold
                else "no"
            )

        return "no"


# ---------------------------------------------------------------------------
# Pipeline construction helpers
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Build an isolated Settings instance rooted in a temporary directory.

    Args:
        tmp_path: Temporary directory that holds uploads and the vector store.
        **overrides: Additional Settings fields to override.

    Returns:
        Configured Settings instance.
    """
    values: dict[str, Any] = {
        "UPLOAD_DIR": Path(tmp_path) / "uploads",
        "CHROMA_DIR": Path(tmp_path) / "chroma",
        "OLLAMA_MODEL": "test-model",
        "TOP_K_RESULTS": 5,
    }
    values.update(overrides)
    return Settings(**values)


def make_pipeline(
    tmp_path: Path,
    llm_client: LLMClient | None = None,
    embedder: Any | None = None,
    **settings_overrides: Any,
) -> TriagePipeline:
    """Build a hermetic pipeline with no network dependency.

    Args:
        tmp_path: Temporary directory for uploads and vector storage.
        llm_client: LLM double; defaults to a grounded analysis double.
        embedder: Embedding engine; defaults to the deterministic offline embedder.
        **settings_overrides: Settings overrides.

    Returns:
        A ready TriagePipeline.
    """
    config = make_settings(tmp_path, **settings_overrides)
    embedding_engine = embedder if embedder is not None else OfflineHashingEmbedder()
    vector_store = VectorStore(
        persist_dir=config.CHROMA_DIR,
        embedding_engine=embedding_engine,
    )
    return TriagePipeline(
        config=config,
        embedding_engine=embedding_engine,
        vector_store=vector_store,
        llm_client=llm_client or GroundedAnalysisLLM(),
    )


def stage_text(
    pipeline: TriagePipeline,
    document_id: str,
    text: str,
    filename: str = "contract.txt",
) -> Path:
    """Write a document body into the pipeline's upload directory.

    Args:
        pipeline: Target pipeline.
        document_id: Document identifier the file will be ingested under.
        text: Document body.
        filename: Filename to store the body as.

    Returns:
        Path written.
    """
    upload_dir = Path(pipeline.config.UPLOAD_DIR) / document_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / filename
    path.write_text(text, encoding="utf-8")
    return path


async def ingest_text_async(
    pipeline: TriagePipeline,
    document_id: str,
    text: str,
    filename: str = "contract.txt",
) -> Any:
    """Stage and ingest a text document from asynchronous code.

    Args:
        pipeline: Target pipeline.
        document_id: Document identifier to index under.
        text: Document body.
        filename: Filename to store the body as.

    Returns:
        The UploadResponse produced by ingestion.
    """
    path = stage_text(pipeline, document_id, text, filename)
    return await pipeline.ingest_document(file_path=path, document_id=document_id)


def ingest_text(
    pipeline: TriagePipeline,
    document_id: str,
    text: str,
    filename: str = "contract.txt",
) -> tuple[Path, Any]:
    """Stage and ingest a text document from synchronous code.

    Args:
        pipeline: Target pipeline.
        document_id: Document identifier to index under.
        text: Document body.
        filename: Filename to store the body as.

    Returns:
        Tuple of (path written, UploadResponse).
    """
    import asyncio

    path = stage_text(pipeline, document_id, text, filename)
    response = asyncio.run(pipeline.ingest_document(file_path=path, document_id=document_id))
    return path, response


def ingest_sample_document(
    pipeline: TriagePipeline,
    document_id: str,
    sample_filename: str,
) -> Any:
    """Ingest one of the committed sample contracts under a stable identifier.

    Args:
        pipeline: Target pipeline.
        document_id: Document identifier to index under.
        sample_filename: File name inside ``tests/sample_documents/``.

    Returns:
        The UploadResponse produced by ingestion.
    """
    _, response = ingest_text(
        pipeline,
        document_id,
        (SAMPLE_DIR / sample_filename).read_text(encoding="utf-8"),
        filename=sample_filename,
    )
    return response


async def ingest_sample_document_async(
    pipeline: TriagePipeline,
    document_id: str,
    sample_filename: str,
) -> Any:
    """Asynchronous variant of :func:`ingest_sample_document`.

    Args:
        pipeline: Target pipeline.
        document_id: Document identifier to index under.
        sample_filename: File name inside ``tests/sample_documents/``.

    Returns:
        The UploadResponse produced by ingestion.
    """
    return await ingest_text_async(
        pipeline,
        document_id,
        (SAMPLE_DIR / sample_filename).read_text(encoding="utf-8"),
        filename=sample_filename,
    )


def run_async(coro: Any) -> Any:
    """Run a coroutine from synchronous test code.

    Args:
        coro: Coroutine to execute.

    Returns:
        The coroutine's result.
    """
    import asyncio

    return asyncio.run(coro)


def unique_threads() -> int:
    """Return the current thread count, used by leak-style assertions.

    Returns:
        Number of active threads.
    """
    return threading.active_count()


def normalized(text: str) -> str:
    """Expose the canonical matching form for assertions in tests.

    Args:
        text: Input text.

    Returns:
        Canonicalized lowercase text.
    """
    return normalize_for_match(text)
