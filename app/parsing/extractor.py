"""Document text extraction and sentence-aware chunking for legal files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pdfplumber

from app.models import DocumentChunk


def extract_text_from_pdf(file_path: Path) -> list[dict[str, Any]]:
    """Extract page-level text and layout information from a PDF file using pdfplumber.

    Args:
        file_path: Path to the target PDF file.

    Returns:
        A list of dictionaries, one per page, containing:
            - page_number (1-indexed int)
            - text (extracted string)
            - layout_info (dictionary of layout metadata)
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {file_path}")

    pages_data: list[dict[str, Any]] = []

    with pdfplumber.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            layout_info: dict[str, Any] = {
                "width": float(page.width),
                "height": float(page.height),
                "page_number": page_idx,
                "char_count": len(text),
                "has_tables": len(page.find_tables()) > 0,
            }
            pages_data.append(
                {
                    "page_number": page_idx,
                    "text": text,
                    "layout_info": layout_info,
                    "source_file": file_path.name,
                }
            )

    return pages_data


def extract_text_from_txt(file_path: Path) -> list[dict[str, Any]]:
    """Extract text and simulated page metadata from a plain text file.

    Handles pagination if form feed ('\x0c') characters are present, otherwise
    treats the entire content as page 1.

    Args:
        file_path: Path to the target text file.

    Returns:
        A list of dictionaries representing document pages.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Text file does not exist: {file_path}")

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = file_path.read_text(encoding="latin-1", errors="replace")

    raw_pages = content.split("\x0c") if "\x0c" in content else [content]
    pages_data: list[dict[str, Any]] = []

    for idx, page_text in enumerate(raw_pages, start=1):
        layout_info: dict[str, Any] = {
            "page_number": idx,
            "char_count": len(page_text),
            "format": "txt",
        }
        pages_data.append(
            {
                "page_number": idx,
                "text": page_text,
                "layout_info": layout_info,
                "source_file": file_path.name,
            }
        )

    return pages_data


def _split_into_sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Split text into sentences while tracking exact character offsets.

    Uses regex matching typical sentence boundaries (periods, question marks,
    exclamations followed by whitespace or double newlines) while preserving
    legal references and numbering.

    Args:
        text: Input string.

    Returns:
        List of tuples: (start_char, end_char, sentence_substring).
    """
    if not text:
        return []

    # Matches sentence boundary:
    # 1. Period, question mark, or exclamation followed by whitespace or newline
    # 2. Double newlines (paragraph boundary)
    # 3. Trailing text up to end of string
    sentence_pattern = re.compile(
        r".+?(?:(?<=[.!?])\s+|[\r\n]{2,}|\Z)",
        re.DOTALL,
    )

    spans: list[tuple[int, int, str]] = []
    for match in sentence_pattern.finditer(text):
        start = match.start()
        end = match.end()
        raw_text = match.group(0)
        if raw_text:
            spans.append((start, end, raw_text))

    return spans


def chunk_document(
    pages: list[dict[str, Any]],
    chunk_size: int = 1000,
    overlap: int = 200,
    source_file: str = "",
) -> list[DocumentChunk]:
    """Sliding window chunker that respects sentence boundaries and tracks source offsets.

    Each generated chunk includes exact character offsets (start_char, end_char)
    within the source page, enabling precise verbatim citation tracing.

    Args:
        pages: List of dictionaries with 'page_number', 'text', and optional 'source_file'.
        chunk_size: Target maximum chunk length in characters.
        overlap: Desired character overlap between adjacent chunks.
        source_file: Default fallback filename for chunks.

    Returns:
        A list of DocumentChunk instances with sequential chunk indices.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size")

    chunks: list[DocumentChunk] = []
    global_chunk_idx = 0

    for page_dict in pages:
        page_number: int = page_dict.get("page_number", 1)
        page_text: str = page_dict.get("text", "")
        file_name: str = page_dict.get("source_file", source_file) or source_file or "unknown"

        if not page_text or not page_text.strip():
            continue

        spans = _split_into_sentence_spans(page_text)
        if not spans:
            continue

        i = 0
        total_spans = len(spans)

        while i < total_spans:
            # Handle exceptionally large individual sentences exceeding chunk_size
            span_start, span_end, span_text = spans[i]
            if (span_end - span_start) > chunk_size:
                # Subdivide oversized span into chunk_size pieces with overlap
                sub_start = span_start
                while sub_start < span_end:
                    sub_end = min(sub_start + chunk_size, span_end)
                    content = page_text[sub_start:sub_end]
                    if content.strip():
                        chunks.append(
                            DocumentChunk(
                                content=content,
                                page_number=page_number,
                                chunk_index=global_chunk_idx,
                                start_char=sub_start,
                                end_char=sub_end,
                                source_file=file_name,
                            )
                        )
                        global_chunk_idx += 1
                    if sub_end == span_end:
                        break
                    sub_start += chunk_size - overlap
                i += 1
                continue

            current_start = span_start
            current_end = span_end
            j = i

            # Accumulate sentences until chunk_size is reached
            while j + 1 < total_spans:
                next_end = spans[j + 1][1]
                if next_end - current_start <= chunk_size:
                    current_end = next_end
                    j += 1
                else:
                    break

            content = page_text[current_start:current_end]
            if content.strip():
                chunks.append(
                    DocumentChunk(
                        content=content,
                        page_number=page_number,
                        chunk_index=global_chunk_idx,
                        start_char=current_start,
                        end_char=current_end,
                        source_file=file_name,
                    )
                )
                global_chunk_idx += 1

            if j + 1 >= total_spans:
                break

            # Calculate rollback index for next chunk to satisfy overlap
            next_i = j + 1
            if overlap > 0:
                accum_overlap = 0
                k = j
                while k > i:
                    span_len = spans[k][1] - spans[k][0]
                    accum_overlap += span_len
                    if accum_overlap >= overlap:
                        next_i = k
                        break
                    k -= 1

            # Ensure forward progress
            if next_i <= i:
                next_i = i + 1

            i = next_i

    return chunks


def extract_and_chunk(
    file_path: Path,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> tuple[list[DocumentChunk], int]:
    """Extract text from a legal document and partition it into sentence-aware chunks.

    Dispatches extraction logic based on file suffix (.pdf, .txt, .md, .text).

    Args:
        file_path: Absolute or relative Path to the target document.
        chunk_size: Target maximum chunk length in characters.
        overlap: Character overlap between consecutive chunks.

    Returns:
        A tuple of (list of DocumentChunk, total page count).

    Raises:
        ValueError: If the file extension is unsupported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = extract_text_from_pdf(path)
    elif suffix in (".txt", ".md", ".text"):
        pages = extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported formats: .pdf, .txt, .md")

    total_pages = len(pages)
    chunks = chunk_document(
        pages=pages,
        chunk_size=chunk_size,
        overlap=overlap,
        source_file=path.name,
    )
    return chunks, total_pages
