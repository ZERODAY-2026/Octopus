"""Main RAG triage pipeline for legal document analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import shutil
from typing import Any

from app.config import Settings, settings
from app.embeddings.engine import EmbeddingEngine
from app.llm.inference import LLMClient
from app.models import (
    CitedPassage,
    ClauseAnalysis,
    DEFAULT_LEGAL_DISCLAIMER,
    DocumentChunk,
    DocumentSummary,
    QueryResponse,
    RetrievedChunk,
    UploadResponse, CrossDocRow, CrossDocResponse, DocumentInfo,
)
from app.parsing.abstention import resolve_answer_status
from app.parsing.citations import SourceSpan, snap_citation
from app.parsing.clauses import normalize_clause_types
from app.parsing.extractor import extract_and_chunk
from app.prompts.legal import (
    LEGAL_DISCLAIMER,
    QUERY_TEMPLATE,
    SUMMARY_TEMPLATE,
    SYSTEM_PROMPT, CROSSDOC_TEMPLATE,
)
from app.retrieval import is_verdict_question, VERDICT_RESPONSE_PREFIX, unknown_entity_phrases
from app.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)

#: Retrieval labels this module injects into the prompt. A language model
#: occasionally echoes them back into the answer, where they are not part of the
#: document and would be scored as ungrounded text.
_CONTEXT_MARKER_RE = re.compile(r"\[\s*EXCERPT\s*\d+[^\]]*\]", re.IGNORECASE)

#: Sentences the model writes to cite its own retrieval context ("this is stated
#: in excerpts 3.3"). They reference excerpt numbering, not the document, so they
#: cannot be grounded and are removed deterministically.
_META_REFERENCE_RE = re.compile(
    r"this (?:is|was) "
    r"(?:stated|outlined|mentioned|noted|found|described|detailed|supported|covered)"
    r"\s+in\s+(?:excerpts?|sections?|the context)\b",
    re.IGNORECASE,
)

#: Sentence boundary used only for dropping self-citation sentences. Splits on
#: terminal punctuation followed by whitespace, so periods inside references such
#: as "3.3" do not end a sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

#: Distinctive fragments of the system prompt. A document can contain text that
#: tries to make the model echo its own instructions -- the payload in
#: ``tests/sample_documents/adversarial_injection.txt`` does exactly that. When
#: one of these appears in a reply, the reply is discarded rather than shown.
_SYSTEM_PROMPT_ECHO_MARKERS: tuple[str, ...] = (
    "critical operational rules",
    "verbatim citation requirement",
    "you are an expert ai legal document analysis",
    "strict factual grounding",
    "separation of fact and interpretation",
    "professional legal escalation",
)


def _leaks_system_prompt(text: str) -> bool:
    """Detect a reply that is repeating the assistant's own instructions.

    Args:
        text: Model-generated answer or summary.

    Returns:
        True when a distinctive system-prompt fragment is present.
    """
    normalized = (text or "").lower()
    return any(marker in normalized for marker in _SYSTEM_PROMPT_ECHO_MARKERS)



def _sanitize_answer_text(text: str) -> str:
    """Strip prompt artifacts from model prose so it reads as document text.

    The context blocks handed to the model are labelled with retrieval metadata.
    Left in place, those labels are neither document content nor grounded, so
    they inflate the apparent answer while adding nothing a reader can verify.
    Removing them deterministically keeps the answer limited to statements the
    citation layer can actually check.

    Args:
        text: Raw answer or summary text from the model.

    Returns:
        The text with excerpt markers and self-citation sentences removed.
    """
    if not text:
        return text

    without_markers = _CONTEXT_MARKER_RE.sub("", text)

    kept: list[str] = []
    for fragment in _SENTENCE_SPLIT_RE.split(without_markers):
        candidate = fragment.strip()
        if not candidate:
            continue
        if _META_REFERENCE_RE.search(candidate):
            continue
        kept.append(candidate)

    cleaned = " ".join(kept) if kept else without_markers.strip()
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+([.,;!?])", r"\1", cleaned)
    return cleaned.strip()


def _parse_llm_json(raw_text: str) -> dict[str, Any] | None:
    """Safely extract and parse JSON object from LLM response text.

    Handles raw JSON, markdown-wrapped JSON code blocks, and responses
    with leading/trailing explanatory text.

    Args:
        raw_text: Raw string output from LLM.

    Returns:
        Parsed dictionary if successful, or None if no valid JSON object found.
    """
    if not raw_text or not raw_text.strip():
        return None

    cleaned = raw_text.strip()

    # 1. Direct JSON parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2. Extract markdown fenced code block: ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # 3. Search for outermost matching braces
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


def _build_clause_analyses(raw_clauses: list[Any]) -> list[ClauseAnalysis]:
    """Normalize and validate a list of clause dictionaries into ClauseAnalysis models.

    Args:
        raw_clauses: Raw items returned by LLM under 'clauses' or 'key_clauses'.

    Returns:
        List of validated ClauseAnalysis instances.
    """
    clauses: list[ClauseAnalysis] = []
    for item in raw_clauses:
        if not isinstance(item, dict):
            continue

        raw_passages = item.get("cited_passages") or []
        passages: list[CitedPassage] = []
        for p in raw_passages:
            if isinstance(p, dict):
                text = str(p.get("text", "")).strip()
                if not text:
                    continue
                try:
                    page_num = max(1, int(p.get("page_number") or 1))
                except (ValueError, TypeError):
                    page_num = 1
                try:
                    start_c = max(0, int(p.get("start_char") or 0))
                except (ValueError, TypeError):
                    start_c = 0
                try:
                    end_c = max(0, int(p.get("end_char") or 0))
                except (ValueError, TypeError):
                    end_c = 0

                passages.append(
                    CitedPassage(
                        text=text,
                        page_number=page_num,
                        start_char=start_c,
                        end_char=end_c,
                    )
                )
            elif isinstance(p, str) and p.strip():
                passages.append(
                    CitedPassage(
                        text=p.strip(),
                        page_number=1,
                        start_char=0,
                        end_char=0,
                    )
                )

        confidence = item.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"

        clause_type = str(item.get("clause_type") or "other").strip()
        summary = _sanitize_answer_text(
            str(item.get("plain_language_summary") or "").strip()
        )

        for canonical_type in normalize_clause_types([clause_type]) or ["other"]:
            clauses.append(
                ClauseAnalysis(
                    clause_type=canonical_type,
                    plain_language_summary=summary,
                    cited_passages=passages,
                    confidence=confidence,
                )
            )
    return clauses


def _spans_from_results(results: list[dict[str, Any]]) -> list[SourceSpan]:
    """Build citation source spans from raw vector-store hits.

    Args:
        results: Raw hits from ``VectorStore.query``, each with ``content`` and
            ``metadata`` (``page_number``/``start_char``/``end_char``).

    Returns:
        List of SourceSpan instances, skipping hits with empty content.
    """
    spans: list[SourceSpan] = []

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    for item in results:
        content = str(item.get("content") or "")
        if not content.strip():
            continue
        meta = item.get("metadata", {}) or {}
        spans.append(
            SourceSpan(
                content=content,
                page_number=max(1, _as_int(meta.get("page_number") or 1, 1)),
                start_char=max(0, _as_int(meta.get("start_char") or 0, 0)),
                end_char=max(0, _as_int(meta.get("end_char") or 0, 0)),
            )
        )
    return spans


def _snap_clause_citations(
    clauses: list[ClauseAnalysis],
    spans: list[SourceSpan],
) -> tuple[list[ClauseAnalysis], list[str], int]:
    """Replace model quotations with exact, located spans of the source text.

    A language model asked to quote a contract quotes approximately: it
    truncates mid-sentence and occasionally puts a refusal or a paraphrase into
    the citation field. Trusting that text would make the project's promise that
    every summary maps back to a *verifiable quotation* probabilistic.

    Instead, each candidate quote is resolved against the retrieved source
    (see :mod:`app.parsing.citations`). A located quote is replaced by the
    document's own characters with the document's own offsets; a quote that
    cannot be located is discarded and reported to the caller, so the user is
    told the text could not be verified rather than shown an unverifiable
    passage dressed up as a source.

    Args:
        clauses: Clause analyses whose ``cited_passages`` hold model quotations.
        spans: Source spans retrieved for this query, with real page/offset data.

    Returns:
        Tuple of (clauses with anchored citations, list of rejected quotations,
        number of clauses dropped for having no verifiable citation at all).
    """
    anchored: list[ClauseAnalysis] = []
    rejected: list[str] = []
    dropped_clauses = 0

    for clause in clauses:
        passages: list[CitedPassage] = []
        for passage in clause.cited_passages:
            match = snap_citation(passage.text, spans)
            if match is None:
                rejected.append(passage.text)
                continue
            passages.append(
                CitedPassage(
                    text=match.text,
                    page_number=match.page_number,
                    start_char=match.start_char,
                    end_char=match.end_char,
                )
            )

        # A clause is a *verifiable finding* only if at least one quotation could
        # be located in the source. Presenting a clause whose every quote failed
        # verification would break the promise that each summary maps back to a
        # real passage, so the clause is withheld rather than trusted.
        if not passages:
            dropped_clauses += 1
            continue

        anchored.append(
            ClauseAnalysis(
                clause_type=clause.clause_type,
                plain_language_summary=clause.plain_language_summary,
                cited_passages=passages,
                confidence=clause.confidence,
            )
        )

    return anchored, rejected, dropped_clauses


_GOVERNING_LAW_RE = re.compile(
    r"governed by[^.]{0,60}?laws? of (?:the )?(?:State of |Commonwealth of )?([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)",
    re.IGNORECASE,
)


def _jurisdiction_flags(document_text: str) -> tuple[list[str], str]:
    """Flag the governing law (if the text names one) and build a specific escalation."""
    match = _GOVERNING_LAW_RE.search(re.sub(r"\s+", " ", document_text))
    if not match:
        return (
            ["The document does not name a governing law; which jurisdiction's rules apply is unclear."],
            "Ask a lawyer licensed where you live or operate which law would apply, and have them review this agreement.",
        )
    place = match.group(1).strip()
    return (
        [f"The document says it is governed by the law of {place}; whether a term is enforceable depends on that law and on the facts."],
        f"Have a lawyer qualified in {place} law review the clauses you are relying on before you sign or act.",
    )


def _clean_query(question: str, unknown: list[str], document_text: str) -> str:
    """Drop words of names that are not in the document so they do not skew search."""
    vocab = set(re.findall(r"[a-z0-9]+", document_text.lower()))
    cleaned = question
    for phrase in unknown:
        kept = [w for w in phrase.split() if re.sub(r"\W", "", w).lower() in vocab]
        cleaned = cleaned.replace(phrase, " ".join(kept))
    return cleaned


class TriagePipeline:
    """Core RAG orchestration pipeline for legal document parsing, embedding, querying, and summarization."""

    def __init__(
        self,
        config: Settings | None = None,
        embedding_engine: EmbeddingEngine | None = None,
        vector_store: VectorStore | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Initialize the triage pipeline with components configured for offline execution.

        Args:
            config: Application settings instance; defaults to global settings.
            embedding_engine: Pre-initialized EmbeddingEngine, or None to create one.
            vector_store: Pre-initialized VectorStore, or None to create one.
            llm_client: Pre-initialized LLMClient, or None to create one.
        """
        self.config = config or settings
        self.embedding_engine = embedding_engine or EmbeddingEngine(
            model_name=self.config.EMBEDDING_MODEL
        )
        self.vector_store = vector_store or VectorStore(
            persist_dir=self.config.CHROMA_DIR,
            embedding_engine=self.embedding_engine,
        )
        self.llm_client = llm_client or LLMClient(
            base_url=self.config.OLLAMA_BASE_URL,
            model_name=self.config.OLLAMA_MODEL,
        )

    async def ingest_document(self, file_path: Path, document_id: str) -> UploadResponse:
        """Extract, chunk, and index a legal document into the local vector store.

        Args:
            file_path: Path to the target PDF or text document.
            document_id: Unique identifier for the document collection.

        Returns:
            UploadResponse containing ingestion statistics and document metadata.
        """
        path = Path(file_path)
        chunks, num_pages = extract_and_chunk(
            file_path=path,
            chunk_size=self.config.CHUNK_SIZE,
            overlap=self.config.CHUNK_OVERLAP,
        )

        self.vector_store.add_document(document_id=document_id, chunks=chunks)

        logger.info(
            "Ingested document '%s' (id: %s): %d chunks across %d pages",
            path.name,
            document_id,
            len(chunks),
            num_pages,
        )

        return UploadResponse(
            document_id=document_id,
            filename=path.name,
            num_chunks=len(chunks),
            num_pages=num_pages,
        )

    async def query_document(self, document_id: str, question: str) -> QueryResponse:
        """Query the document using semantic search and analyze retrieved clauses with LLM.

        Args:
            document_id: Target document identifier.
            question: User inquiry regarding the document.

        Returns:
            QueryResponse with synthesized answer, classified clauses, citations, and disclaimer.
        """
        document_text = self.vector_store.get_text(document_id)
        unknown = unknown_entity_phrases(question, document_text)
        results = self.vector_store.query(
            document_id=document_id,
            query_text=_clean_query(question, unknown, document_text),
            n_results=self.config.TOP_K_RESULTS,
        )
        return await self._synthesize(
            question=question, results=results, document_text=document_text, unknown_entities=unknown
        )

    def _to_retrieved_chunk(self, result: dict[str, Any]) -> RetrievedChunk:
        """Convert a raw vector-store hit into a validated RetrievedChunk.

        Args:
            result: Raw dictionary from ``VectorStore.query``.

        Returns:
            Populated RetrievedChunk carrying citation metadata.
        """
        meta = result.get("metadata", {}) or {}

        def _as_int(key: str, default: int) -> int:
            try:
                return int(meta.get(key) or default)
            except (TypeError, ValueError):
                return default

        distance = result.get("distance")

        return RetrievedChunk(
            chunk_id=str(result.get("chunk_id") or ""),
            content=str(result.get("content") or ""),
            page_number=max(1, _as_int("page_number", 1)),
            chunk_index=max(0, _as_int("chunk_index", 0)),
            start_char=max(0, _as_int("start_char", 0)),
            end_char=max(0, _as_int("end_char", 0)),
            source_file=str(meta.get("source_file") or ""),
            distance=float(distance) if isinstance(distance, (int, float)) else None,
        )

    def retrieve(
        self,
        document_id: str,
        question: str,
        n_results: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve ranked evidence chunks without invoking the LLM.

        This is the retrieval half of the RAG pipeline, exposed so evaluation can
        score retrieval quality (recall@k, precision@k, MRR, nDCG) independently
        of generation quality.

        Args:
            document_id: Target document identifier.
            question: Natural language query.
            n_results: Number of chunks to fetch; defaults to ``TOP_K_RESULTS``.

        Returns:
            Ranked list of RetrievedChunk objects, best match first.
        """
        raw_results = self.vector_store.query(
            document_id=document_id,
            query_text=question,
            n_results=n_results or self.config.TOP_K_RESULTS,
        )
        return [self._to_retrieved_chunk(item) for item in raw_results]

    async def query_document_with_evidence(
        self, document_id: str, question: str
    ) -> tuple[QueryResponse, list[RetrievedChunk]]:
        """Answer a question and also return the exact evidence that was used.

        Retrieval runs once, so the returned evidence is guaranteed to be the
        same context the answer was synthesized from.

        Args:
            document_id: Target document identifier.
            question: Natural language question.

        Returns:
            Tuple of (QueryResponse, ranked evidence chunks).
        """
        document_text = self.vector_store.get_text(document_id)
        unknown = unknown_entity_phrases(question, document_text)
        raw_results = self.vector_store.query(
            document_id=document_id,
            query_text=_clean_query(question, unknown, document_text),
            n_results=self.config.TOP_K_RESULTS,
        )
        response = await self._synthesize(
            question=question, results=raw_results, document_text=document_text, unknown_entities=unknown
        )
        return response, [self._to_retrieved_chunk(item) for item in raw_results]

    async def _synthesize(
        self,
        question: str,
        results: list[dict[str, Any]],
        document_text: str = "",
        unknown_entities: list[str] | None = None,
    ) -> QueryResponse:
        """Turn retrieved excerpts into a cited, disclaimer-bearing answer.

        Args:
            question: Natural language question.
            results: Raw vector-store hits, best match first.

        Returns:
            QueryResponse with answer, clause analyses, citations and notes.
        """
        if not results:
            return QueryResponse(
                answer="The document does not contain provisions addressing this inquiry, or the document is empty.",
                answer_status="silent",
                clauses=[],
                disclaimer=LEGAL_DISCLAIMER,
                uncertainty_notes=[
                    "The document is silent on this matter.",
                    "No relevant excerpts matching the query were found in the document.",
                ],
            )

        context_blocks: list[str] = []
        for idx, item in enumerate(results, start=1):
            content = item.get("content", "")
            meta = item.get("metadata", {}) or {}
            page_num = meta.get("page_number", 1)
            start_char = meta.get("start_char", 0)
            end_char = meta.get("end_char", 0)
            context_blocks.append(
                f"[EXCERPT {idx} | Page: {page_num} | Char offset: {start_char}-{end_char}]\n{content}"
            )

        context_text = "\n\n".join(context_blocks)
        prompt = (
            QUERY_TEMPLATE.replace("{context}", context_text)
            .replace("{question}", question)
        )

        try:
            raw_response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1,
            )
        except Exception as err:
            logger.error("LLM inference failed for query: %s", err)
            return QueryResponse(
                answer=f"Inference error: unable to contact local LLM ({err}).",
                answer_status="silent",
                clauses=[],
                disclaimer=LEGAL_DISCLAIMER,
                uncertainty_notes=[
                    "Local inference service connection error.",
                    f"Retrieved {len(results)} candidate excerpts from local database.",
                ],
            )

        parsed = _parse_llm_json(raw_response)

        if parsed is not None:
            answer = _sanitize_answer_text(str(parsed.get("answer") or "").strip())
            if not answer:
                answer = raw_response.strip()

            clauses = _build_clause_analyses(parsed.get("clauses") or [])
            clauses, rejected_quotes, dropped_clauses = _snap_clause_citations(
                clauses, _spans_from_results(results)
            )

            raw_notes = parsed.get("uncertainty_notes") or []
            if isinstance(raw_notes, list):
                uncertainty_notes = [str(n) for n in raw_notes if n]
            elif isinstance(raw_notes, str) and raw_notes.strip():
                uncertainty_notes = [raw_notes.strip()]
            else:
                uncertainty_notes = []

            if rejected_quotes:
                logger.warning(
                    "Withheld %d citation(s) that could not be verified against the source "
                    "(count=%d; text redacted)",
                    len(rejected_quotes),
                    len(rejected_quotes),
                )
                uncertainty_notes.append(
                    f"{len(rejected_quotes)} quotation(s) returned by the model could not be "
                    "verified verbatim against the document excerpts and were withheld; those "
                    "claims remain unverified and the document may be silent on them."
                )

            if dropped_clauses:
                logger.warning(
                    "Withheld %d clause(s) with no verifiable citation", dropped_clauses
                )
                uncertainty_notes.append(
                    f"{dropped_clauses} clause(s) could not be supported by any verifiable "
                    "quotation from the document and were not reported as findings."
                )

            answer_status = resolve_answer_status(
                answer, model_status=parsed.get("answer_status")
            )

            # Withhold verified clauses only for a full silence. Partial answers
            # (e.g. "X is Y. The excerpts do not specify Z.") must keep the
            # clauses that support what the document does say.
            if clauses and answer_status == "silent":
                logger.info(
                    "Withheld %d clause finding(s): answer_status=silent",
                    len(clauses),
                )
                uncertainty_notes.append(
                    f"The excerpts do not answer this question, so the {len(clauses)} "
                    "clause(s) the model detected nearby are not reported as findings; "
                    "they concern other matters and must not be read as an answer."
                )
                clauses = []

            for name in unknown_entities or []:
                uncertainty_notes.append(
                    f"'{name}' does not appear exactly in this document, so the answer "
                    "is based on the agreement actually uploaded; check that it is the one you mean."
                )

            verdict_declined = is_verdict_question(question)
            jurisdiction_flags: list[str] = []
            escalation = ""
            if verdict_declined:
                if not answer.startswith(VERDICT_RESPONSE_PREFIX):
                    answer = VERDICT_RESPONSE_PREFIX + "Here is what the text itself says: " + answer
                jurisdiction_flags, escalation = _jurisdiction_flags(document_text)

            # The disclaimer is a constant: never accept one from model output,
            # because a document can inject text that rewrites it.
            disclaimer = LEGAL_DISCLAIMER

            if _leaks_system_prompt(answer):
                logger.warning("Discarded an answer that echoed the system instructions")
                answer = (
                    "The model's reply attempted to disclose its internal instructions "
                    "rather than analyse the document, so it was discarded. No summary "
                    "is provided for that request."
                )
                answer_status = "silent"
                uncertainty_notes.append(
                    "The document contains text that tries to redirect the analysis "
                    "assistant; such instructions are treated as document content and "
                    "are not followed."
                )

            return QueryResponse(
                answer=answer,
                answer_status=answer_status,
                clauses=clauses,
                disclaimer=disclaimer,
                uncertainty_notes=uncertainty_notes,
            )

        # Fallback if LLM output could not be parsed as JSON
        logger.warning("LLM response did not contain valid JSON; falling back to plain text extraction")
        fallback_answer = raw_response.strip()
        return QueryResponse(
            answer=fallback_answer,
            answer_status=resolve_answer_status(fallback_answer),
            clauses=[],
            disclaimer=LEGAL_DISCLAIMER,
            uncertainty_notes=[
                "Automated analysis note: Model returned unstructured output; structured clause parsing could not be verified.",
            ],
        )

    def list_document_infos(self) -> list[DocumentInfo]:
        """Document ids with their original filenames (for the UI selector)."""
        return [
            DocumentInfo(document_id=d, filename=self.vector_store.get_filename(d))
            for d in self.list_documents()
        ]

    async def query_across_documents(
            self, question: str, document_ids: list[str] | None = None
    ) -> CrossDocResponse:
        """Ask the same question of every selected document and tabulate verified answers.

        Each document is judged separately from its own best excerpts, so a
        document is reported "yes" only when a quotation that really occurs in
        that document supports it. Documents are never ranked by similarity
        alone, and the response states how many documents were checked.
        """
        ids = document_ids or self.list_documents()
        rows: list[CrossDocRow] = []
        for doc_id in ids:
            filename = self.vector_store.get_filename(doc_id)
            hits = self.vector_store.query(doc_id, question, n_results=4)
            if not hits:
                rows.append(CrossDocRow(document_id=doc_id, filename=filename, verdict="unclear",
                                        explanation="No indexed text for this document."))
                continue
            context = "\n\n".join(
                f"[EXCERPT {i}]\n{h['content']}" for i, h in enumerate(hits, start=1)
            )
            prompt = (
                CROSSDOC_TEMPLATE.replace("{filename}", filename)
                .replace("{context}", context)
                .replace("{question}", question)
            )
            try:
                raw = await self.llm_client.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.0)
            except Exception as err:  # keep going: one failure must not hide the others
                logger.error("Cross-document inference failed for %s: %s", doc_id, err)
                rows.append(CrossDocRow(document_id=doc_id, filename=filename, verdict="unclear",
                                        explanation="Local model call failed for this document."))
                continue
            parsed = _parse_llm_json(raw) or {}
            verdict = str(parsed.get("verdict") or "unclear").strip().lower()
            if verdict not in ("yes", "no", "unclear"):
                verdict = "unclear"
            evidence = None
            quote = str(parsed.get("evidence_quote") or "").strip()
            if quote:
                match = snap_citation(quote, _spans_from_results(hits))
                if match is not None:
                    evidence = CitedPassage(text=match.text, page_number=match.page_number,
                                            start_char=match.start_char, end_char=match.end_char)
            if verdict == "yes" and evidence is None:
                verdict = "unclear"  # a yes without a verifiable quote is not a finding
            rows.append(CrossDocRow(
                document_id=doc_id, filename=filename, verdict=verdict,
                explanation=_sanitize_answer_text(str(parsed.get("explanation") or "")),
                evidence=evidence,
            ))
        yes = [r.filename for r in rows if r.verdict == "yes"]
        summary = (
                f"Checked {len(rows)} document(s). Verified yes: {len(yes)}"
                + (f" ({', '.join(yes)})." if yes else ".")
        )
        return CrossDocResponse(question=question, rows=rows, documents_checked=len(rows), summary=summary)

    async def summarize_document(self, document_id: str) -> DocumentSummary:
        """Generate a structured legal summary of the document and its key provisions.

        Args:
            document_id: Target document identifier.

        Returns:
            DocumentSummary containing an executive summary, key clauses, and disclaimer.
        """
        collection_name = self.vector_store._sanitize_collection_name(document_id)

        try:
            collection = self.vector_store.client.get_collection(
                name=collection_name,
                embedding_function=self.vector_store.embedding_function,
            )
            data = collection.get(include=["documents", "metadatas"])
        except Exception as err:
            logger.warning("Could not load collection for summary '%s': %s", collection_name, err)
            data = {}

        docs: list[str] = data.get("documents") or []
        metas: list[dict[str, Any]] = data.get("metadatas") or []

        filename = f"{document_id}.pdf"
        if metas and isinstance(metas[0], dict) and "source_file" in metas[0]:
            filename = str(metas[0]["source_file"])

        items: list[tuple[int, str, dict[str, Any]]] = []
        for doc, meta in zip(docs, metas):
            idx = meta.get("chunk_index", 0) if isinstance(meta, dict) else 0
            items.append((idx, doc, meta if isinstance(meta, dict) else {}))
        items.sort(key=lambda x: x[0])

        if not items:
            return DocumentSummary(
                document_id=document_id,
                filename=filename,
                key_clauses=[],
                overall_summary="The document is empty or has no indexed content.",
                disclaimer=LEGAL_DISCLAIMER,
            )

        # Select a representative sample of chunks to stay within context window
        max_chunks = 15
        if len(items) > max_chunks:
            step = max(1, len(items) // max_chunks)
            sampled = items[::step][:max_chunks]
        else:
            sampled = items

        context_blocks: list[str] = []
        for num, (_, doc, meta) in enumerate(sampled, start=1):
            page = meta.get("page_number", 1)
            start = meta.get("start_char", 0)
            end = meta.get("end_char", 0)
            context_blocks.append(
                f"[EXCERPT {num} | Page: {page} | Char offset: {start}-{end}]\n{doc}"
            )

        context_text = "\n\n".join(context_blocks)
        prompt = (
            SUMMARY_TEMPLATE.replace("{filename}", filename)
            .replace("{context}", context_text)
        )

        try:
            raw_response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1,
            )
        except Exception as err:
            logger.error("LLM inference failed for document summary: %s", err)
            return DocumentSummary(
                document_id=document_id,
                filename=filename,
                key_clauses=[],
                overall_summary=f"Inference error: unable to contact local LLM ({err}).",
                disclaimer=LEGAL_DISCLAIMER,
            )

        parsed = _parse_llm_json(raw_response)
        if parsed is not None:
            overall_summary = _sanitize_answer_text(
                str(parsed.get("overall_summary") or "").strip()
            )
            if not overall_summary:
                overall_summary = raw_response.strip()

            key_clauses = _build_clause_analyses(parsed.get("key_clauses") or [])
            key_clauses, rejected_quotes, dropped_clauses = _snap_clause_citations(
                key_clauses,
                _spans_from_results(
                    [{"content": doc, "metadata": meta} for _, doc, meta in sampled]
                ),
            )
            if rejected_quotes or dropped_clauses:
                logger.warning(
                    "Summary withheld %d unverifiable citation(s) and %d uncited clause(s)",
                    len(rejected_quotes),
                    dropped_clauses,
                )

            return DocumentSummary(
                document_id=document_id,
                filename=filename,
                key_clauses=key_clauses,
                overall_summary=overall_summary,
                disclaimer=LEGAL_DISCLAIMER,
            )

        # Fallback if unstructured text was returned
        return DocumentSummary(
            document_id=document_id,
            filename=filename,
            key_clauses=[],
            overall_summary=raw_response.strip(),
            disclaimer=LEGAL_DISCLAIMER,
        )

    async def delete_document(self, document_id: str) -> bool:
        """Delete document records from the vector store and clear uploaded files.

        Args:
            document_id: Target document identifier.

        Returns:
            True if deletion was initiated successfully.
        """
        try:
            self.vector_store.delete_document(document_id)
        except Exception as err:
            logger.warning("Error deleting vector store collection for '%s': %s", document_id, err)

        upload_subdir = Path(self.config.UPLOAD_DIR) / document_id
        if upload_subdir.exists():
            shutil.rmtree(upload_subdir, ignore_errors=True)

        return True

    def list_documents(self) -> list[str]:
        """List all document IDs currently indexed or uploaded.

        Returns:
            Alphabetically sorted list of document ID strings.
        """
        doc_ids: set[str] = set()

        try:
            collections = self.vector_store.client.list_collections()
            for col in collections:
                meta = getattr(col, "metadata", None) or {}
                orig_id = meta.get("original_document_id")
                if orig_id:
                    doc_ids.add(orig_id)
                else:
                    doc_ids.add(col.name)
        except Exception as err:
            logger.warning("Failed listing vector store collections: %s", err)

        upload_dir = Path(self.config.UPLOAD_DIR)
        if upload_dir.is_dir():
            for item in upload_dir.iterdir():
                if item.is_dir():
                    doc_ids.add(item.name)

        return sorted(list(doc_ids))
