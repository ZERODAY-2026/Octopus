# ⚖️ Contracter — Private Legal Document Triage Agent

> **⚠️ IMPORTANT DISCLAIMER:** Contracter is a document analysis tool, **NOT** a legal
> service. It does not provide legal advice. All outputs are automated summaries and
> should be verified by a qualified legal professional before making any decisions.

**Contracter** is a fully offline, privacy-first legal document triage agent that
processes contracts on your local machine. It reads PDF and TXT files, extracts key
clauses, summarizes them in plain language, and maps every summary back to exact quoted
passages in the source document — all without any data ever leaving your device.

---

## 🔒 Privacy Guarantee

Contracter enforces **zero-network privacy** by design:

| Component | Runs Where | Network Access |
|-----------|-----------|----------------|
| LLM (Gemma 3n E4B) | Ollama on localhost | ❌ None |
| Embeddings (all-MiniLM-L6-v2) | Local GPU/CPU | ❌ None |
| Vector DB (ChromaDB) | Local filesystem | ❌ None |
| Document Storage | Local `data/` directory | ❌ None |
| FastAPI Backend | localhost:8000 | ❌ None |
| Streamlit Frontend | localhost:8501 | ❌ None |

- **No telemetry**, no analytics, no cloud calls.
- Private document contents, embeddings, and query history never leave your machine.
- Validated by automated tests that disable system network interfaces.

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────┐
│                   Streamlit UI (:8501)                 │
│  ┌─────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Upload  │  │ Ask Questions│  │ Quick Analysis  │  │
│  └────┬────┘  └──────┬───────┘  └────────┬────────┘  │
└───────┼──────────────┼───────────────────┼────────────┘
        │              │                   │
        ▼              ▼                   ▼
┌───────────────────────────────────────────────────────┐
│                 FastAPI Backend (:8000)                │
│  ┌──────────────────────────────────────────────────┐ │
│  │              Triage Pipeline (RAG)                │ │
│  │                                                   │ │
│  │  PDF/TXT ──► Chunk ──► Embed ──► Store ──► Query │ │
│  │  (pdfplumber) (sliding)  (ST)   (Chroma) (Ollama)│ │
│  └──────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
        │              │                   │
        ▼              ▼                   ▼
   ┌─────────┐  ┌───────────┐     ┌──────────────┐
   │ChromaDB │  │ Sentence  │     │    Ollama     │
   │(local   │  │Transformers│    │ Gemma 3n E4B │
   │  disk)  │  │  (GPU)    │     │   (GPU)      │
   └─────────┘  └───────────┘     └──────────────┘
```

---

## ✨ Features

### Core Analysis
- **Clause Extraction**: Automatically identifies obligations, deadlines, penalties,
  and termination conditions, mapped onto one canonical clause vocabulary.
- **Plain-Language Summaries**: Complex legal language translated into readable English.
- **Citation Tracking**: Every summary maps to exact quoted passages with page number
  and character offset references.
- **Confidence Scoring**: Each finding tagged as High, Medium, or Low confidence.

### Calibrated Uncertainty
- Explicitly states when a document **is silent** on a queried topic.
- Refuses to invent or speculate on unsupported conclusions.
- Withholds clause findings when the answer itself is a refusal, so a finding can
  never be presented beside a "the document is silent" statement.
- Example: *"The contract specifies a 30-day notice period but is silent on whether
  notice must be delivered by email or physical letter."*

### Verifiable, Anchored Citations
- Every quotation is located back in the source and re-emitted as the document's
  own characters, with the source's own page and offset references.
- Quotations the model cannot reproduce are withheld, and clauses with no
  verifiable citation are not reported as findings.

### Offline RAG Evaluation
- A RAGAS-style evaluation harness scores groundedness, answer relevancy, context
  relevance/precision/recall, answer correctness, retrieval quality, citation
  accuracy and abstention behaviour.
- Runs in judged mode against the local model, or in a deterministic lexical mode
  that needs no model and can run inside the network-disabled privacy tests.

### Legal Disclaimer System
- All outputs clearly separate **explicit document statements** from **interpretations**.
- Persistent disclaimer recommending consultation with qualified legal counsel.
- System prompt enforces this behavior at the LLM level.

---

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1060 (6GB) | NVIDIA RTX 5070 Ti (12GB) |
| CPU | Intel Core i7 | Intel Core Ultra 9 |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB free | 50 GB free |
| OS | Linux (tested on Fedora) | Linux with NVIDIA drivers |

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.14+** with [uv](https://docs.astral.sh/uv/) package manager
2. **Ollama** installed and running ([install guide](https://ollama.com/download))
3. **NVIDIA drivers** and **CUDA toolkit** for GPU acceleration

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/contracter.git
cd contracter

# Pull the Gemma 3n E4B model
ollama pull gemma3n:e4b

# Install dependencies
uv sync

# Create data directories
mkdir -p data/uploads data/chroma_db
```

### Running

```bash
# Terminal 1: Start the FastAPI backend
uv run python main.py

# Terminal 2: Start the Streamlit frontend
uv run streamlit run streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

### First Use

1. Upload a PDF or TXT contract via the sidebar
2. Wait for document processing (parsing → chunking → embedding → storing)
3. Ask questions like:
   - *"What are the termination conditions?"*
   - *"What deadlines does this contract specify?"*
   - *"What penalties exist for breach of contract?"*
4. Use the **Quick Analysis** tab for one-click extraction of key clause types

---

## 🧪 Testing

### Run All Tests

```bash
uv run pytest tests/ -v
```

The default suite is **hermetic**: no network, no model downloads and no Ollama.
It uses a deterministic offline embedder and scripted model doubles, so it runs
in a bare CI container in a few seconds.

Markers select the tests that need real resources:

```bash
uv run pytest -m embedding   # requires the cached sentence-transformers model
uv run pytest -m ollama      # requires a reachable local Ollama server
uv run pytest -m privacy     # the zero-network validation suite
uv run pytest -m "not slow"  # fast unit tests only
```

### Test Modules

| Module | What it covers |
|--------|----------------|
| `tests/test_metrics.py` | Metric maths: faithfulness, relevancy, recall/precision, MRR, nDCG, citation checks. |
| `tests/test_citations.py` | Verbatim citation anchoring, repaired matches, rejection of unverifiable quotes. |
| `tests/test_privacy.py` | The whole pipeline with the network disabled; proves zero outbound attempts. |
| `tests/test_retrieval_integration.py` | CUAD-style retrieval, chunk provenance, clause mapping and the citation guarantee. |
| `tests/test_adversarial.py` | Malformed input, vague contracts, prompt injection and system-prompt disclosure. |
| `tests/test_clauses.py` | Canonical clause-type normalization   (aliases, compound labels, one finding per category, withholding findings on a refusal). |
| `tests/test_evaluation_harness.py` | Threshold selection, aggregation, reporting and dataset loading. |

### Privacy Validation

Verifies the entire pipeline works with the network disabled, at both the test
level and as a standalone artifact you can run and demonstrate:

```bash
uv run pytest tests/test_privacy.py -v
uv run python scripts/verify_offline.py            # real embedder + local model
uv run python scripts/verify_offline.py --no-real-model --no-llm
```

The standalone script parses, chunks, embeds, indexes, retrieves, answers,
summarizes and deletes a document with every outbound socket call intercepted,
then reports how many non-loopback connections were attempted (target: zero).

---

## 📊 RAG Evaluation

`scripts/run_evaluation.py` replays a CUAD-style ground-truth dataset through the
pipeline and scores it, writing a JSON report alongside a Markdown rendering. It
exits non-zero when a metric falls below its threshold, so it doubles as a CI
gate.

```bash
uv run python scripts/run_evaluation.py                          # deterministic lexical mode
uv run python scripts/run_evaluation.py --embedder minilm        # production embedder
uv run python scripts/run_evaluation.py --judge --model gemma3n:e4b
```

The metric set follows the RAGAS paper (*Es et al., arXiv:2309.15217*) together
with the standard retrieval metrics:

| Metric | Question it answers |
|--------|---------------------|
| `faithfulness` (groundedness) | Is every claim in the answer supported by the retrieved excerpts? |
| `answer_relevancy` | Does the answer actually address the question? |
| `context_relevance` | Are the retrieved excerpts on topic? |
| `context_precision` | Are the relevant excerpts ranked first? |
| `context_recall` | Did retrieval cover the reference answer? |
| `answer_correctness` | Precision / recall / F1 against the reference answer. |
| `retrieval_hit_rate`, `retrieval_recall`, `retrieval_precision`, `retrieval_mrr`, `retrieval_ndcg` | Is the retrieval half of the pipeline healthy? |
| `citation_grounding_rate`, `citation_locator_accuracy`, `citation_offset_accuracy` | Do citations appear verbatim, on the right page, at the right offsets? |
| `abstention_recall`, `over_abstention_rate`, `fabricated_clause_rate` | Is calibrated uncertainty working? |

There are two execution modes. **Judged mode** (`--judge`) uses the local Ollama
model to decompose answers into claims and verify them, exactly as RAGAS does.
**Lexical mode** is a deterministic token/number-overlap approximation that needs
no model, so it runs in CI and inside the network-disabled privacy tests. Each
mode has its own documented threshold set; both compute and report *every*
metric, and each threshold carries a comment recording why it sits where it does.

### Threshold Calibration

Thresholds are targets for a competent *local* pipeline, not the headline
numbers from papers that assume a frontier judge. Three are deliberately looser
than the literature, each for a structural reason recorded beside the constant:

- `answer_relevancy` → **0.65** judged. RAGAS itself cites ~0.75 with a strong
  judge, but here a 4B model writes the reverse question and the score is an
  embedding distance.
- `context_relevance` → **0.30** judged. This is a per-chunk mean, so with
  `k=5` retrieved excerpts and ~2 relevant clauses the arithmetic ceiling is
  ~0.4 however good retrieval is; `retrieval_precision@R` above is the metric
  that actually isolates retrieval quality.
- `answer_clause_type_accuracy` → **0.45**. The metric is recall over the clause
  categories a question touches, and a question spanning a notice *deadline* and
  the *termination* right it serves is credited half when the model reports one
  category. The metric's maths is asserted directly by unit tests.

When the evaluator is handed a non-semantic (bag-of-words) embedder,
`answer_relevancy` is still computed and reported but **never gated**, because
such an embedder cannot recognise that a correct paraphrase is on topic.

### Canonical Clause Vocabulary

A model labels clauses inconsistently — `"obligation | penalty | termination"`,
`"term_and_termination"`, `"Limitation of Liability"` all describe one finding
worth of text. `app/parsing/clauses.py` owns a single canonical vocabulary,
splits compound labels, folds aliases onto their category and emits one finding
per category, so a compound label can no longer read as one unrecognized string
in the interface *or* score as a miss in evaluation.

### Verifiable Citations by Construction

A language model asked to quote a contract quotes approximately: it truncates
mid-sentence and sometimes puts a refusal or paraphrase in the citation field.
Rather than trust the model, `app/parsing/citations.py` locates each generated
quote inside the retrieved source and replaces it with the document's *own*
characters and offsets. Quotes that cannot be located are dropped, a clause with
no verifiable citation is withheld, and the user is told the text could not be
verified — so the "every summary maps back to a quotation" promise is structural,
not probabilistic.

---

## 📡 API Reference

### `POST /api/upload`
Upload a PDF or TXT document for analysis.

**Request:** `multipart/form-data` with `file` field  
**Response:**
```json
{
  "document_id": "abc123-...",
  "filename": "contract.pdf",
  "num_chunks": 42,
  "num_pages": 8
}
```

### `POST /api/query`
Ask a question about an uploaded document.

**Request:**
```json
{
  "document_id": "abc123-...",
  "question": "What are the termination conditions?"
}
```

**Response:**
```json
{
  "answer": "The contract can be terminated with 30 days written notice...",
  "clauses": [
    {
      "clause_type": "termination",
      "plain_language_summary": "Either party may terminate with 30 days notice.",
      "cited_passages": [
        {
          "text": "Either Party may terminate this Agreement upon thirty (30) days...",
          "page_number": 4,
          "start_char": 1250,
          "end_char": 1380
        }
      ],
      "confidence": "high"
    }
  ],
  "disclaimer": "This analysis is not legal advice...",
  "uncertainty_notes": ["The contract does not specify the method of notice delivery."]
}
```

### `GET /api/documents`
List all uploaded documents.

### `GET /api/documents/{document_id}/summary`
Get a full structured summary of a document.

### `DELETE /api/documents/{document_id}`
Remove a document and its data.

### `GET /api/health`
Check system health (Ollama, ChromaDB, GPU status).

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Ollama + Gemma 3n E4B | Local inference on GPU |
| Backend | Python 3.14 + FastAPI | REST API server |
| Frontend | Streamlit | Interactive web interface |
| Vector DB | ChromaDB | Local persistent storage |
| Embeddings | sentence-transformers | On-device embedding generation |
| PDF Parsing | pdfplumber | Text and layout extraction |
| Testing | pytest | Unit, integration, and security tests |

---

## 📁 Project Structure

```
contracter/
├── main.py                    # FastAPI application entry point
├── streamlit_app.py           # Streamlit frontend
├── pyproject.toml             # Dependencies and project config
├── README.md                  # This file
├── CHANGELOG.md               # Release notes
├── LICENSE                    # MIT License
├── app/
│   ├── config.py              # Application settings
│   ├── models.py              # Pydantic data models
│   ├── parsing/
│   │   ├── extractor.py       # PDF/TXT extraction and chunking
│   │   ├── citations.py       # Verbatim citation anchoring
│   │   ├── clauses.py         # Canonical clause-type vocabulary
│   │   └── abstention.py      # Calibrated-uncertainty detection
│   ├── embeddings/
│   │   ├── engine.py          # sentence-transformers engine
│   │   └── deterministic.py   # Offline hashing embedder (tests/CI)
│   ├── vectorstore/
│   │   └── store.py           # ChromaDB local storage
│   ├── llm/
│   │   └── inference.py       # Ollama LLM client
│   ├── prompts/
│   │   └── legal.py           # System prompts and templates
│   ├── pipeline/
│   │   └── triage.py          # RAG triage pipeline
│   ├── evaluation/            # Offline RAG evaluation harness
│   │   ├── metrics.py         # RAGAS-style + retrieval + citation metrics
│   │   ├── dataset.py         # CUAD-style sample models and loaders
│   │   ├── runner.py          # Scoring, aggregation and reports
│   │   ├── prompts.py         # LLM-judge prompt templates
│   │   └── text.py            # Deterministic text utilities
│   └── api/
│       └── routes.py          # FastAPI routes
├── scripts/
│   ├── verify_offline.py      # Zero-network privacy validation artifact
│   └── run_evaluation.py      # RAG evaluation CLI / CI gate
├── tests/
│   ├── conftest.py            # Shared test fixtures
│   ├── support.py             # Network blocker, model doubles, helpers
│   ├── test_metrics.py        # Metric maths
│   ├── test_citations.py      # Citation anchoring
│   ├── test_privacy.py        # Offline network-disabled tests
│   ├── test_retrieval_integration.py  # CUAD-style integration tests
│   ├── test_adversarial.py    # Adversarial security tests
│   ├── test_clauses.py        # Clause-type normalization and withholding
│   ├── test_evaluation_harness.py     # Harness and reporting tests
│   └── sample_documents/
│       ├── ground_truth.jsonl # CUAD-style evaluation dataset
│       └── generate_test_pdfs.py
└── data/
    ├── uploads/               # User uploaded documents
    └── chroma_db/             # ChromaDB persistent storage
```

---

## 🤝 Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `uv run pytest tests/ -v`
2. New features include corresponding tests
3. Privacy constraints are maintained (no network calls)
4. The legal disclaimer system is preserved in all outputs

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

> **Remember:** This tool assists with document analysis. It is **not** a substitute
> for qualified legal counsel. Always consult a licensed attorney for important legal
> decisions.
