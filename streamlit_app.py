"""Contracter — Private Legal Document Triage Frontend.

A full-featured Streamlit interface for private, local legal document analysis,
clause classification, citation verification, and uncertainty tracking.
All processing runs locally with on-device models and local vector database.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import html
import logging
from typing import Any

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
API_BASE_URL: str = "http://127.0.0.1:8000"

DEFAULT_DISCLAIMER: str = (
    "This tool provides document analysis assistance only. It does NOT "
    "constitute legal advice. Always consult a qualified legal professional "
    "for important legal decisions."
)

logger = logging.getLogger("contracter_frontend")

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Contracter — Private Legal Document Triage",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS Styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Main typography and headers */
.main-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1.0rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}

/* Sidebar Disclaimer Box */
.sidebar-disclaimer-box {
    background-color: rgba(239, 68, 68, 0.08);
    border: 2px solid #ef4444;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 18px;
}
.sidebar-disclaimer-heading {
    font-weight: 700;
    font-size: 0.85rem;
    color: #b91c1c;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.sidebar-disclaimer-body {
    font-size: 0.80rem;
    line-height: 1.45;
    color: #991b1b;
    margin: 0;
}

/* System Status Badges */
.status-pill-online {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: rgba(16, 185, 129, 0.12);
    border: 1px solid #10b981;
    color: #065f46;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.80rem;
    font-weight: 600;
}
.status-pill-offline {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: rgba(239, 68, 68, 0.12);
    border: 1px solid #ef4444;
    color: #991b1b;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.80rem;
    font-weight: 600;
}

/* Clause Badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 700;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-right: 8px;
    vertical-align: middle;
}
.badge-obligation {
    background-color: #e0f2fe;
    color: #0369a1;
    border: 1px solid #7dd3fc;
}
.badge-deadline {
    background-color: #fef3c7;
    color: #b45309;
    border: 1px solid #fcd34d;
}
.badge-penalty {
    background-color: #fee2e2;
    color: #b91c1c;
    border: 1px solid #fca5a5;
}
.badge-termination {
    background-color: #f3e8ff;
    color: #6d28d9;
    border: 1px solid #d8b4fe;
}
.badge-confidentiality {
    background-color: #ede9fe;
    color: #5b21b6;
    border: 1px solid #c4b5fd;
}
.badge-liability {
    background-color: #ffedd5;
    color: #c2410c;
    border: 1px solid #fed7aa;
}
.badge-dispute {
    background-color: #f1f5f9;
    color: #334155;
    border: 1px solid #cbd5e1;
}
.badge-other {
    background-color: #f3f4f6;
    color: #4b5563;
    border: 1px solid #d1d5db;
}

/* Confidence Indicator Badges */
.conf-badge {
    display: inline-block;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 4px;
    vertical-align: middle;
}
.conf-high {
    background-color: #dcfce7;
    color: #15803d;
    border: 1px solid #86efac;
}
.conf-medium {
    background-color: #fef3c7;
    color: #b45309;
    border: 1px solid #fcd34d;
}
.conf-low {
    background-color: #fee2e2;
    color: #b91c1c;
    border: 1px solid #fca5a5;
}

/* Blockquote Citation Box */
.citation-box {
    border-left: 4px solid #3b82f6;
    background-color: rgba(59, 130, 246, 0.05);
    padding: 10px 14px;
    margin: 8px 0 12px 0;
    border-radius: 0 6px 6px 0;
    font-family: Georgia, Cambria, "Times New Roman", serif;
    font-style: italic;
    color: #1e293b;
    line-height: 1.55;
}
.citation-meta {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-style: normal;
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    margin-top: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Uncertainty and Silence Section */
.uncertainty-box {
    background-color: rgba(245, 158, 11, 0.08);
    border: 1px solid #fde68a;
    border-left: 4px solid #d97706;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 14px 0;
    color: #92400e;
}
.uncertainty-title {
    font-weight: 700;
    font-size: 0.90rem;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Synthesis Answer Container */
.answer-container {
    background-color: rgba(241, 245, 249, 0.7);
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 20px;
    color: #0f172a;
    font-size: 1.02rem;
    line-height: 1.65;
}

/* Disclaimer Footer */
.disclaimer-footer {
    background-color: rgba(239, 68, 68, 0.06);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #991b1b;
    margin-top: 26px;
    line-height: 1.45;
}

/* Active document header pill */
.doc-pill {
    display: inline-block;
    background-color: #e2e8f0;
    color: #334155;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.85rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Async HTTP API Client
# ---------------------------------------------------------------------------
class BackendApiClient:
    """Asynchronous HTTP client communicating with the Contracter FastAPI backend."""

    def __init__(self, base_url: str = API_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    async def get_health(self) -> dict[str, Any]:
        """Fetch system health and component status."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=4.0) as client:
            resp = await client.get("/api/health")
            resp.raise_for_status()
            return resp.json()

    async def get_documents(self) -> list[Any]:
        """Fetch list of all ingested legal documents."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=8.0) as client:
            resp = await client.get("/api/documents")
            resp.raise_for_status()
            return resp.json()

    async def upload_document(
        self,
        filename: str,
        content: bytes,
        mime_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        """Upload a PDF or TXT contract file for chunking and vector indexing."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=120.0) as client:
            files = {"file": (filename, content, mime_type)}
            resp = await client.post("/api/upload", files=files)
            resp.raise_for_status()
            return resp.json()

    async def query_document(self, document_id: str, question: str) -> dict[str, Any]:
        """Query an ingested contract using RAG pipeline with citation extraction."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=120.0) as client:
            payload = {"document_id": document_id, "question": question}
            resp = await client.post("/api/query", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def query_all(self, question: str, document_ids: list[str] | None) -> dict[str, Any]:
        """Ask one question of several documents and get a verified per-document table."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=600.0) as client:
            resp = await client.post(
                "/api/query-all", json={"question": question, "document_ids": document_ids}
            )
            resp.raise_for_status()
            return resp.json()

    async def get_document_details(self) -> list[dict[str, Any]]:
        """Document ids with original filenames."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=30.0) as client:
            resp = await client.get("/api/documents-details")
            resp.raise_for_status()
            return resp.json()

    async def get_document_summary(self, document_id: str) -> dict[str, Any]:
        """Generate or retrieve a full structured summary of the specified document."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=180.0) as client:
            resp = await client.get(f"/api/documents/{document_id}/summary")
            resp.raise_for_status()
            return resp.json()

    async def delete_document(self, document_id: str) -> dict[str, Any]:
        """Delete an ingested document and its vector collection."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=15.0) as client:
            resp = await client.delete(f"/api/documents/{document_id}")
            resp.raise_for_status()
            return resp.json()


api_client = BackendApiClient()


def run_async_call(coroutine):
    """Safely execute an asynchronous coroutine within Streamlit's sync execution loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coroutine).result()
        else:
            return loop.run_until_complete(coroutine)
    except RuntimeError:
        return asyncio.run(coroutine)


# ---------------------------------------------------------------------------
# Helper Formatters & Badge Renderers
# ---------------------------------------------------------------------------
def get_clause_badge_html(clause_type: str) -> str:
    """Return an HTML badge element styled according to the clause category."""
    ctype = (clause_type or "").lower().strip()
    label = clause_type.title() if clause_type else "General"

    if "obligation" in ctype:
        css_class = "badge-obligation"
        icon = "📌"
        label = "Obligation"
    elif "deadline" in ctype or "date" in ctype or "time" in ctype or "milestone" in ctype:
        css_class = "badge-deadline"
        icon = "⏱️"
        label = "Deadline"
    elif "penalty" in ctype or "liquidat" in ctype or "fine" in ctype or "remedy" in ctype:
        css_class = "badge-penalty"
        icon = "⚠️"
        label = "Penalty"
    elif "termination" in ctype or "cancel" in ctype or "survival" in ctype:
        css_class = "badge-termination"
        icon = "🚪"
        label = "Termination"
    elif "confidential" in ctype or "nda" in ctype:
        css_class = "badge-confidentiality"
        icon = "🔒"
        label = "Confidentiality"
    elif "liab" in ctype or "indemn" in ctype:
        css_class = "badge-liability"
        icon = "🛡️"
        label = "Liability / Indemnity"
    elif "dispute" in ctype or "law" in ctype or "jurisdiction" in ctype:
        css_class = "badge-dispute"
        icon = "⚖️"
        label = "Dispute Resolution"
    else:
        css_class = "badge-other"
        icon = "📄"

    return f'<span class="badge {css_class}">{icon} {html.escape(label)}</span>'


def get_confidence_badge_html(confidence: str) -> str:
    """Return an HTML badge element color-coded by confidence level."""
    conf = (confidence or "medium").lower().strip()
    if conf == "high":
        css_class = "conf-high"
        label = "● High Confidence"
    elif conf == "low":
        css_class = "conf-low"
        label = "● Low Confidence"
    else:
        css_class = "conf-medium"
        label = "● Medium Confidence"

    return f'<span class="conf-badge {css_class}">{html.escape(label)}</span>'


def render_clause_card(clause: dict[str, Any], index: int) -> None:
    """Render a structured expandable section for a single identified clause."""
    clause_type = clause.get("clause_type", "General")
    summary = clause.get("plain_language_summary", "")
    citations = clause.get("cited_passages", [])
    confidence = clause.get("confidence", "medium")

    type_badge = get_clause_badge_html(clause_type)
    conf_badge = get_confidence_badge_html(confidence)

    title_preview = summary[:95] + ("..." if len(summary) > 95 else "")
    if not title_preview.strip():
        title_preview = f"Clause #{index + 1}"

    expander_title = f"{clause_type.title()} — {title_preview}"
    with st.expander(expander_title, expanded=(index == 0)):
        st.markdown(f'<div style="margin-bottom: 12px;">{type_badge} {conf_badge}</div>', unsafe_allow_html=True)

        st.markdown("**Plain Language Summary:**")
        if summary:
            st.markdown(f"> {html.escape(summary)}")
        else:
            st.caption("No plain language summary provided.")

        st.markdown(f"**Verbatim Cited Passages ({len(citations)}):**")
        if citations:
            for cit in citations:
                cit_text = cit.get("text", "").strip()
                page_no = cit.get("page_number", 1)
                start_c = cit.get("start_char", 0)
                end_c = cit.get("end_char", 0)

                escaped_text = html.escape(cit_text).replace("\n", "<br>")
                st.markdown(
                    f"""
                    <div class="citation-box">
                        "{escaped_text}"
                        <div class="citation-meta">
                            <span>📄 <strong>Page {page_no}</strong></span>
                            <span>•</span>
                            <span>Character Range: <code>[{start_c} : {end_c}]</code></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No verbatim citation passages recorded for this clause.")


def render_structured_result(data: dict[str, Any]) -> None:
    """Render the full structured analysis: Answer, Clauses, Uncertainty Notes, and Disclaimer."""
    answer = data.get("answer", "")
    answer_status = str(data.get("answer_status") or "answered").strip().lower()
    clauses = data.get("clauses") or data.get("key_clauses") or []
    uncertainty_notes = data.get("uncertainty_notes", [])
    disclaimer = data.get("disclaimer", DEFAULT_DISCLAIMER)

    status_labels = {
        "answered": ("Answered", "The document covers this question."),
        "partial": (
            "Partial answer",
            "The document answers part of the question and is silent on another part.",
        ),
        "silent": ("Document silent", "The document does not address this question."),
    }
    status_title, status_help = status_labels.get(
        answer_status, status_labels["answered"]
    )
    st.caption(f"Answer status: **{status_title}** — {status_help}")

    # Partial answers are rendered as two explicit parts so the silence note
    # sits alongside the finding instead of replacing it.
    if answer_status == "partial":
        st.markdown("#### What the document says")
        if answer:
            st.markdown(
                f'<div class="answer-container">{html.escape(answer)}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f"#### Identified clauses ({len(clauses)})")
        if clauses:
            for idx, clause in enumerate(clauses):
                render_clause_card(clause, idx)
        else:
            st.info("No verified contractual clauses were retained for this query.")

        st.markdown("#### What the document does not say")
        if uncertainty_notes:
            notes_list = "".join(
                f"<li>{html.escape(str(note))}</li>"
                for note in uncertainty_notes
                if str(note).strip()
            )
            if notes_list:
                st.markdown(
                    f"""
                    <div class="uncertainty-box">
                        <div class="uncertainty-title">Matters the document is silent or ambiguous on</div>
                        <ul style="margin: 0; padding-left: 20px;">
                            {notes_list}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No additional silence notes were recorded.")
    else:
        # 1. Plain language synthesis / answer
        if answer:
            st.markdown("#### Plain language synthesis")
            st.markdown(
                f'<div class="answer-container">{html.escape(answer)}</div>',
                unsafe_allow_html=True,
            )

        # 2. Key Clauses Breakdown
        st.markdown(f"#### Identified clauses ({len(clauses)})")
        if clauses:
            for idx, clause in enumerate(clauses):
                render_clause_card(clause, idx)
        else:
            st.info(
                "No specific contractual clauses were detected or extracted for this query."
            )

        # 3. Uncertainty Notes / Silence Section
        st.markdown("#### Document silence and uncertainty notes")
        if uncertainty_notes:
            notes_list = "".join(
                f"<li>{html.escape(str(note))}</li>"
                for note in uncertainty_notes
                if str(note).strip()
            )
            if notes_list:
                st.markdown(
                    f"""
                    <div class="uncertainty-box">
                        <div class="uncertainty-title">Matters the document is silent or ambiguous on</div>
                        <ul style="margin: 0; padding-left: 20px;">
                            {notes_list}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.caption(
                    "The document explicitly covers the subject; no omissions noted."
                )
        else:
            st.caption(
                "The document explicitly addresses the queried points; "
                "no specific omissions or silence noted."
            )

    for flag in data.get("jurisdiction_flags") or []:
        st.warning(f"Jurisdiction: {flag}")
    if data.get("escalation"):
        st.info(f"Suggested next step: {data['escalation']}")

    excerpts = data.get("retrieved_excerpts") or []
    if excerpts:
        with st.expander(f"Evidence: the {len(excerpts)} excerpts the answer was based on"):
            for i, ex in enumerate(excerpts, start=1):
                st.markdown(f"**Excerpt {i}** - page {ex.get('page_number', '?')}")
                st.code(ex.get("content", ""), language=None)

    # 4. Prominent Legal Disclaimer
    if disclaimer:
        st.markdown(
            f"""
            <div class="disclaimer-footer">
                <strong>Legal notice:</strong> {html.escape(disclaimer)}
            </div>
            """,
            unsafe_allow_html=True,
        )


def normalize_documents_list(raw_response: Any) -> list[dict[str, str]]:
    """Normalize various backend /api/documents formats into uniform dictionaries."""
    docs: list[dict[str, str]] = []
    if isinstance(raw_response, dict):
        raw_response = raw_response.get("documents", [])

    if isinstance(raw_response, list):
        for item in raw_response:
            if isinstance(item, str):
                docs.append({"document_id": item, "filename": item})
            elif isinstance(item, dict):
                doc_id = item.get("document_id") or item.get("id") or item.get("name") or "unknown"
                fname = item.get("filename") or item.get("name") or str(doc_id)
                docs.append({"document_id": str(doc_id), "filename": str(fname)})
    return docs


# ---------------------------------------------------------------------------
# Sidebar Setup
# ---------------------------------------------------------------------------
def render_sidebar() -> tuple[str | None, str | None]:
    """Render the sidebar with disclaimer, upload form, doc selector, health, and privacy note."""

    # 1. Prominent Warning / Disclaimer Box at Top of Sidebar
    st.sidebar.markdown(
        """
        <div class="sidebar-disclaimer-box">
            <div class="sidebar-disclaimer-heading">⚠️ Important Notice</div>
            <p class="sidebar-disclaimer-body">
                This tool provides document analysis assistance only. It does <strong>NOT</strong> constitute
                legal advice. Always consult a qualified legal professional for important legal decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. System Health Indicator
    st.sidebar.markdown("### 🖥️ System Health")
    try:
        health_data = run_async_call(api_client.get_health())
        status_text = health_data.get("status", "online") if isinstance(health_data, dict) else "online"
        st.sidebar.markdown(
            f'<div class="status-pill-online">● Backend Online ({html.escape(str(status_text))})</div>',
            unsafe_allow_html=True,
        )
        with st.sidebar.expander("System Info", expanded=False):
            st.json(health_data)
    except Exception:
        st.sidebar.markdown(
            '<div class="status-pill-offline">● Backend Offline</div>',
            unsafe_allow_html=True,
        )
        st.sidebar.warning(
            "FastAPI backend is unreachable at `http://127.0.0.1:8000`. "
            "Please ensure the backend server is running."
        )

    st.sidebar.markdown("---")

    # 3. Document Upload Section
    st.sidebar.markdown("### 📤 Upload Document")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF or TXT contracts",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload one or more contracts (.pdf or .txt) up to 50MB each.",
    )

    if uploaded_files:
        if st.sidebar.button("🚀 Upload & Process", use_container_width=True):
            last_doc_id: str | None = None
            failures: list[str] = []
            with st.sidebar.status("Processing documents...", expanded=True) as status:
                for uploaded_file in uploaded_files:
                    st.write(f"Indexing {uploaded_file.name} ...")
                    try:
                        result = run_async_call(
                            api_client.upload_document(
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "application/octet-stream",
                            )
                        )
                        last_doc_id = result.get("document_id", last_doc_id)
                    except httpx.HTTPStatusError as e:
                        failures.append(
                            f"{uploaded_file.name}: backend error {e.response.status_code}: {e.response.text}")
                    except Exception as e:
                        failures.append(f"{uploaded_file.name}: {e}")
                status.update(
                    label="Done" if not failures else "Finished with errors",
                    state="complete" if not failures else "error",
                    expanded=bool(failures),
                )
            for message in failures:
                st.sidebar.error(message)
            if last_doc_id:
                st.session_state["selected_doc_id"] = last_doc_id
                if not failures:
                    st.rerun()

    st.sidebar.markdown("---")

    # 4. Document Selector Dropdown
    st.sidebar.markdown("### 📚 Select Document")
    doc_options: list[dict[str, str]] = []
    try:
        try:
            doc_options = normalize_documents_list(run_async_call(api_client.get_document_details()))
        except Exception:
            doc_options = normalize_documents_list(run_async_call(api_client.get_documents()))
        st.session_state["all_doc_options"] = doc_options
    except Exception:
        doc_options = []

    selected_doc_id: str | None = None
    selected_doc_name: str | None = None

    if doc_options:
        doc_labels = [f"{d['filename']} ({d['document_id'][:8]}...)" for d in doc_options]
        default_index = 0

        # Maintain selected document across reruns if present
        current_sel = st.session_state.get("selected_doc_id")
        if current_sel:
            for idx, d in enumerate(doc_options):
                if d["document_id"] == current_sel:
                    default_index = idx
                    break

        selected_label = st.sidebar.selectbox(
            "Active Contract",
            options=doc_labels,
            index=default_index,
            label_visibility="collapsed",
        )
        selected_index = doc_labels.index(selected_label)
        selected_doc_id = doc_options[selected_index]["document_id"]
        selected_doc_name = doc_options[selected_index]["filename"]
        st.session_state["selected_doc_id"] = selected_doc_id

        # Quick document actions
        col_ref, col_del = st.sidebar.columns([1, 1])
        with col_ref:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        with col_del:
            if st.button("🗑️ Delete", use_container_width=True):
                try:
                    run_async_call(api_client.delete_document(selected_doc_id))
                    st.sidebar.success("Document deleted.")
                    st.session_state.pop("selected_doc_id", None)
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Delete failed: {e}")
    else:
        st.sidebar.info("No documents uploaded yet. Please upload a PDF or TXT document above.")

    st.sidebar.markdown("---")

    # 5. Privacy Notice Expander
    with st.sidebar.expander("🔒 Privacy Notice", expanded=False):
        st.markdown(
            """
            **Absolute On-Device Privacy:**
            - **100% Local Execution**: All extraction, chunking, and querying runs locally on this system.
            - **No External Network Calls**: Documents, text embeddings, and inference prompts never leave `localhost`.
            - **Local Embeddings**: Computed on-device using local SentenceTransformers (`all-MiniLM-L6-v2`).
            - **Local Inference**: Powered by local Ollama instance (`gemma3n:e4b`).
            - **Persistent Storage**: Vector indices are kept strictly in local ChromaDB storage.
            """
        )

    return selected_doc_id, selected_doc_name


# ---------------------------------------------------------------------------
# Tab 1: Ask Questions
# ---------------------------------------------------------------------------
def render_tab_ask_questions(doc_id: str, doc_name: str) -> None:
    """Tab 1: Interactive Question & Answering with structured triage breakdown."""
    st.markdown("#### 💬 Ask Questions About This Contract")
    st.caption("Ask specific legal or contractual questions. Every answer cites verbatim passages.")

    user_question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the confidentiality obligations and exceptions under this agreement?",
        key="qa_question_input",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        submit_btn = st.button("🔍 Submit Question", type="primary", use_container_width=True)

    session_qa_key = f"qa_result_{doc_id}"

    if submit_btn and user_question.strip():
        with st.spinner("Analyzing contract chunks & extracting cited clauses..."):
            try:
                response = run_async_call(api_client.query_document(doc_id, user_question.strip()))
                st.session_state[session_qa_key] = response
            except httpx.HTTPStatusError as e:
                st.error(f"Backend returned error {e.response.status_code}: {e.response.text}")
            except httpx.ConnectError:
                st.error("Cannot connect to backend at http://127.0.0.1:8000. Is the FastAPI server running?")
            except Exception as e:
                st.error(f"Query analysis failed: {e}")

    # Display results if present
    if session_qa_key in st.session_state:
        result_data = st.session_state[session_qa_key]
        st.markdown("---")
        render_structured_result(result_data)


# ---------------------------------------------------------------------------
# Tab: Across Documents
# ---------------------------------------------------------------------------
def render_tab_across_documents() -> None:
    """Ask one question of many documents, e.g. 'Which contracts mention arbitration in New York?'."""
    st.markdown("#### 🗂️ Ask Across Documents")
    st.caption("Each document is checked separately; a 'yes' always comes with a verbatim quotation.")
    options: list[dict[str, str]] = st.session_state.get("all_doc_options", [])
    if len(options) < 2:
        st.info("Upload two or more documents to ask questions across them.")
        return
    labels = {d["document_id"]: d["filename"] for d in options}
    chosen = st.multiselect(
        "Documents to check (empty = all)",
        options=list(labels),
        format_func=lambda i: labels[i],
    )
    question = st.text_input(
        "Question for every selected document:",
        placeholder="e.g., Does this contract require arbitration in New York?",
        key="cross_question",
    )
    if st.button("🔍 Check documents", type="primary") and question.strip():
        with st.spinner(f"Checking {len(chosen) or len(options)} document(s)..."):
            try:
                st.session_state["cross_result"] = run_async_call(
                    api_client.query_all(question.strip(), chosen or None)
                )
            except Exception as e:
                st.error(f"Cross-document query failed: {e}")
    result = st.session_state.get("cross_result")
    if result:
        st.success(result.get("summary", ""))
        icon = {"yes": "✅ Yes", "no": "❌ No", "unclear": "❔ Unclear"}
        for row in result.get("rows", []):
            with st.expander(f"{icon.get(row['verdict'], row['verdict'])} - {row['filename']}"):
                if row.get("explanation"):
                    st.write(row["explanation"])
                ev = row.get("evidence")
                if ev:
                    st.markdown(f"Evidence (page {ev.get('page_number', '?')}):")
                    st.code(ev.get("text", ""), language=None)
        st.caption(result.get("disclaimer", DEFAULT_DISCLAIMER))


# ---------------------------------------------------------------------------
# Tab 2: Document Summary
# ---------------------------------------------------------------------------
def render_tab_document_summary(doc_id: str, doc_name: str) -> None:
    """Tab 2: Full Document Summary generation with classified clauses & citations."""
    st.markdown("#### 📋 Comprehensive Document Summary")
    st.caption("Generate an executive triage summary classifying all major clauses, commitments, and terms.")

    session_sum_key = f"summary_result_{doc_id}"

    if st.button("📑 Generate Full Document Summary", type="primary"):
        with st.spinner("Synthesizing complete document triage report (this may take 15–30 seconds)..."):
            try:
                summary_data = run_async_call(api_client.get_document_summary(doc_id))
                st.session_state[session_sum_key] = summary_data
            except httpx.HTTPStatusError as e:
                st.error(f"Backend returned error {e.response.status_code}: {e.response.text}")
            except httpx.ConnectError:
                st.error("Cannot connect to backend at http://127.0.0.1:8000. Is the FastAPI server running?")
            except Exception as e:
                st.error(f"Document summarization failed: {e}")

    if session_sum_key in st.session_state:
        sum_data = st.session_state[session_sum_key]
        st.markdown("---")

        overall = sum_data.get("overall_summary", "")
        if overall:
            st.markdown("#### 📖 Executive Summary")
            st.markdown(f'<div class="answer-container">{html.escape(overall)}</div>', unsafe_allow_html=True)

        key_clauses = sum_data.get("key_clauses", [])
        st.markdown(f"#### 🔑 Classified Key Clauses ({len(key_clauses)})")
        if key_clauses:
            for idx, clause in enumerate(key_clauses):
                render_clause_card(clause, idx)
        else:
            st.info("No key clauses categorized for this agreement.")

        uncertainty = sum_data.get("uncertainty_notes", [])
        if uncertainty:
            st.markdown("#### ⚠️ Document Silence & Missing Terms")
            notes_list = "".join(f"<li>{html.escape(str(n))}</li>" for n in uncertainty if str(n).strip())
            if notes_list:
                st.markdown(
                    f"""
                    <div class="uncertainty-box">
                        <div class="uncertainty-title">⚠️ Omission & Silence Observations</div>
                        <ul style="margin: 0; padding-left: 20px;">
                            {notes_list}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        disclaimer = sum_data.get("disclaimer", DEFAULT_DISCLAIMER)
        if disclaimer:
            st.markdown(
                f"""
                <div class="disclaimer-footer">
                    ⚖️ <strong>Legal Notice:</strong> {html.escape(disclaimer)}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Tab 3: Quick Analysis
# ---------------------------------------------------------------------------
def render_tab_quick_analysis(doc_id: str, doc_name: str) -> None:
    """Tab 3: Pre-built quick analysis queries for key legal risk vectors."""
    st.markdown("#### ⚡ Quick Legal Analysis")
    st.caption("Select a pre-built analysis vector to instantly triage key obligations, deadlines, penalties, and termination rules.")

    session_quick_key = f"quick_result_{doc_id}"

    col1, col2 = st.columns(2)
    with col1:
        btn_obligations = st.button("📌 Extract Obligations", use_container_width=True)
        btn_penalties = st.button("⚠️ Identify Penalties", use_container_width=True)
    with col2:
        btn_deadlines = st.button("⏱️ Find Deadlines", use_container_width=True)
        btn_termination = st.button("🚪 Review Termination Conditions", use_container_width=True)

    triggered_query: str | None = None
    query_title: str | None = None

    if btn_obligations:
        triggered_query = (
            "Extract and categorize all obligations, affirmative duties, restrictive covenants, "
            "and deliverables required of each party under this contract."
        )
        query_title = "Obligations Analysis"
    elif btn_deadlines:
        triggered_query = (
            "Identify all deadlines, timeframes, milestone dates, expiration windows, renewal dates, "
            "and notice periods specified in this agreement."
        )
        query_title = "Deadlines & Milestones Analysis"
    elif btn_penalties:
        triggered_query = (
            "Identify and detail all penalties, late fees, liquidated damages, default interest rates, "
            "indemnifications, and financial liabilities in this contract."
        )
        query_title = "Penalties & Liabilities Analysis"
    elif btn_termination:
        triggered_query = (
            "Review and outline all termination conditions, grounds for termination (for cause and convenience), "
            "notice periods, cure rights, and post-termination survival obligations."
        )
        query_title = "Termination Conditions Analysis"

    if triggered_query:
        with st.spinner(f"Running {query_title}..."):
            try:
                resp = run_async_call(api_client.query_document(doc_id, triggered_query))
                st.session_state[session_quick_key] = {"title": query_title, "data": resp}
            except httpx.HTTPStatusError as e:
                st.error(f"Backend returned error {e.response.status_code}: {e.response.text}")
            except httpx.ConnectError:
                st.error("Cannot connect to backend at http://127.0.0.1:8000. Is the FastAPI server running?")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    if session_quick_key in st.session_state:
        stored = st.session_state[session_quick_key]
        st.markdown("---")
        st.markdown(f"### 📊 Results: {stored['title']}")
        render_structured_result(stored["data"])


# ---------------------------------------------------------------------------
# Main Layout Application Entry
# ---------------------------------------------------------------------------
def main() -> None:
    """Application main flow controlling sidebar selection and tab rendering."""

    # Render Sidebar and retrieve active document
    selected_doc_id, selected_doc_name = render_sidebar()

    # Main Area Header
    st.markdown(
        '<div class="main-title">⚖️ Contracter — Private Legal Document Triage</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Local, privacy-preserving legal document triage with verbatim citation tracing and uncertainty calibration.</div>',
        unsafe_allow_html=True,
    )

    # Empty State: No document selected
    if not selected_doc_id:
        st.info("👈 Please upload a PDF or TXT contract in the sidebar, or select an existing document to begin triage.")

        st.markdown("### 🌟 Key Capabilities")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                **🔒 100% Private & Offline**
                - Completely local Ollama inference
                - On-device SentenceTransformers
                - File-based ChromaDB storage
                - Zero third-party cloud data transfer
                """
            )
        with col2:
            st.markdown(
                """
                **🔍 Grounded Verbatim Citations**
                - Strict quote verification
                - Exact page number tracking
                - Character offset range tracing
                - Eliminates hallucinated terms
                """
            )
        with col3:
            st.markdown(
                """
                **⚖️ Calibrated Legal Triage**
                - Categorized clause classification
                - Confidence indicators (High/Med/Low)
                - Document silence & omission detection
                - Prominent legal disclaimers
                """
            )

        st.markdown("---")
        st.caption(
            "Contracter is designed for pre-triage document analysis. It does not replace legal representation."
        )
        return

    # Active Document Information Bar
    col_info, col_id = st.columns([3, 1])
    with col_info:
        st.markdown(
            f"📄 **Active Document:** `{selected_doc_name or selected_doc_id}`"
        )
    with col_id:
        st.markdown(
            f'<div style="text-align: right;"><span class="doc-pill">ID: {selected_doc_id[:12]}...</span></div>',
            unsafe_allow_html=True,
        )

    # Main Area Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Ask Questions", "📋 Document Summary", "⚡ Quick Analysis", "🗂️ Across Documents"]
    )

    with tab1:
        render_tab_ask_questions(selected_doc_id, selected_doc_name or selected_doc_id)

    with tab2:
        render_tab_document_summary(selected_doc_id, selected_doc_name or selected_doc_id)

    with tab3:
        render_tab_quick_analysis(selected_doc_id, selected_doc_name or selected_doc_id)

    with tab4:
        render_tab_across_documents()

if __name__ == "__main__":
    main()
