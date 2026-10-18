"""Generate PDF versions of the plain-text sample contracts.

PDF parsing is exercised with a real PDF so that ``pdfplumber`` extraction,
page numbering, and character offsets are all covered by the tests. The
generated files are written to ``tests/sample_documents/generated/`` which is
git-ignored: they are build artifacts, rebuilt on demand, not fixtures.

Usage:
    python tests/sample_documents/generate_test_pdfs.py
    python tests/sample_documents/generate_test_pdfs.py --out /tmp/opencode/pdfs
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from fpdf import FPDF

SAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SAMPLE_DIR / "generated"

#: Sample documents converted to PDF. The synthetic extras exercise failure
#: modes that a real contract would not contain.
PDF_SOURCES: tuple[str, ...] = (
    "mutual_nda.txt",
    "service_agreement.txt",
    "sparse_agreement.txt",
    "adversarial_injection.txt",
)

#: A paginated document used to prove multi-page extraction and page numbering.
MULTIPAGE_SOURCE = "multipage_contract.txt"


class _ContractPDF(FPDF):
    """Minimal PDF writer with a footer page number."""

    def footer(self) -> None:  # noqa: D102 - fpdf2 hook
        self.set_y(-15)
        self.set_font("Helvetica", size=8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _write_text_pdf(text: str, destination: Path, *, line_height: float = 6.0) -> Path:
    """Render a text block into a paginated PDF file.

    Args:
        text: Source text to render.
        destination: Output PDF path.
        line_height: Line height in millimetres.

    Returns:
        The path written.
    """
    page = _ContractPDF(format="A4")
    page.set_auto_page_break(auto=True, margin=18)
    page.add_page()
    page.set_font("Helvetica", size=11)

    for raw_line in text.splitlines():
        # Sanitize to the Latin-1 core fonts used by the sample PDFs.
        line = (
            raw_line.replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2013", "-")
            .replace("\u2014", "-")
        )
        line = re.sub(r"[^\x00-\xff]", "", line)
        if not line.strip():
            page.ln(line_height)
            continue
        page.multi_cell(0, line_height, line, new_x="LMARGIN", new_y="NEXT")

    destination.parent.mkdir(parents=True, exist_ok=True)
    page.output(str(destination))
    return destination


def generate_sample_pdfs(out_dir: Path = DEFAULT_OUT_DIR) -> list[Path]:
    """Render every sample contract to PDF.

    Args:
        out_dir: Directory that receives the generated PDFs.

    Returns:
        Paths of the generated PDF files, in a stable order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name in PDF_SOURCES:
        source = SAMPLE_DIR / name
        if not source.is_file():
            continue
        text = source.read_text(encoding="utf-8")
        written.append(_write_text_pdf(text, out_dir / f"{source.stem}.pdf"))

    # Build a deliberately multi-page document so page-number assertions bite.
    multipage_text = "\n".join(
        f"{section}. SECTION {section} OBLIGATIONS\n\n"
        f"The Provider shall perform obligation number {section} within "
        f"{section * 10} days of the Effective Date. Either party may terminate "
        f"this Agreement for convenience upon thirty (30) days' prior written "
        f"notice.\n"
        for section in range(1, 26)
    )
    written.append(_write_text_pdf(multipage_text, out_dir / f"{MULTIPAGE_SOURCE[:-4]}.pdf"))

    return written


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args(argv)

    written = generate_sample_pdfs(args.out)
    for path in written:
        print(f"wrote {path}")
    print(f"{len(written)} PDF(s) generated in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
