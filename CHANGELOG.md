# Changelog

All notable changes to Contracter will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Offline RAG Evaluation Harness** (`app/evaluation/`): a RAGAS-style metric
  suite covering faithfulness/groundedness, answer relevancy, context
  relevance/precision/recall, answer correctness, retrieval quality (hit rate,
  recall@k, precision@k, MRR, nDCG@k), citation accuracy and calibrated
  uncertainty. Ships two execution modes: a local-LLM judged mode and a
  deterministic lexical mode that needs no model and runs in CI.
- **Evaluation CLI** (`scripts/run_evaluation.py`): replays a CUAD-style
  ground-truth dataset through the pipeline, writes JSON and Markdown reports,
  and exits non-zero on a threshold miss so it can gate CI.
- **Verifiable citation anchoring** (`app/parsing/citations.py`): generated
  quotations are located inside the retrieved source and replaced with the
  document's own characters and offsets. Unverifiable quotes are withheld, and a
  clause with no verifiable citation is not reported as a finding.
- **Deterministic offline embedder** (`app/embeddings/deterministic.py`): a
  hashing bag-of-words encoder that makes the test suite and privacy
  verification hermetic, with no model download and no network.
- **Privacy-validation script** (`scripts/verify_offline.py`): runs the complete
  pipeline with every outbound socket call intercepted and reports the number of
  non-loopback connection attempts (target: zero).
- **Test suite**: metric-maths tests, citation-anchoring tests, zero-network
  privacy tests, CUAD-style retrieval/clause-mapping integration tests,
  adversarial tests (malformed input, vague contracts, prompt injection,
  system-prompt disclosure) and evaluation-harness tests.
- **Adversarial hardening**: excerpt labels and self-citation sentences are
  stripped from model prose, and a reply that echoes the system instructions is
  discarded rather than shown.

### Changed

- `TriagePipeline` now anchors every emitted citation to the retrieved source and
  records a calibrated-uncertainty note when a quotation or clause could not be
  verified.
- The system prompt explicitly forbids retrieval artifacts (excerpt labels, page
  numbers, offsets) from appearing in answers.
- ChromaDB telemetry is disabled explicitly rather than relying on the installed
  release's default being quiet.

### Fixed

- Faithfulness no longer penalises a calibrated refusal that contains content
  words: a sentence that only reports document silence is not treated as a
  document claim.

## [1.0.0] — 2026-10-18

### Added

- **Offline Legal Document Triage Pipeline**: Complete RAG pipeline for extracting,
  embedding, and querying legal documents entirely on-device.
- **Document Parsing**: PDF and TXT extraction via `pdfplumber` with clause-boundary-aware
  chunking and sliding window overlap.
- **Local Vector Database**: ChromaDB configured for persistent local storage — no cloud,
  no network, no telemetry.
- **On-Device Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`) running on
  NVIDIA RTX GPU via CUDA for hardware-accelerated embedding generation.
- **Local LLM Inference**: Ollama integration with Gemma 3n E4B edge model, configured
  for RTX 5070 Ti GPU acceleration.
- **Citation Tracking**: Every plain-language summary maps back to exact quoted passages
  with page number and character offset references.
- **Clause Classification**: Automatic detection and categorization of obligations,
  deadlines, penalties, and termination conditions.
- **Calibrated Uncertainty**: System explicitly states when documents are silent on a
  queried topic and refuses to invent unsupported conclusions.
- **Legal Disclaimer System**: All outputs clearly separate document statements from
  interpretations and recommend consulting qualified legal counsel.
- **FastAPI Backend**: RESTful API with endpoints for upload, query, summarize, and
  document management.
- **Streamlit Frontend**: Interactive web UI with document upload, question-asking,
  structured summaries, and quick analysis buttons.
- **Privacy Validation Tests**: Automated tests that disable network interfaces and
  verify complete offline functionality.
- **Integration Test Suite**: `pytest` tests using CUAD-style sample documents to
  validate chunk retrieval and clause mapping accuracy.
- **Adversarial Security Tests**: Tests for malformed files, vague clauses, prompt
  injection attempts, path traversal, and SQL injection patterns.

### Security

- Zero-network architecture: document contents, embeddings, and telemetry never leave
  the local machine.
- Input validation on all file uploads (extension, size, path traversal).
- Prompt injection resistance via strict system prompts and document text isolation.
