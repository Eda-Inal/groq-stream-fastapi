from __future__ import annotations

import fitz  # pymupdf


class PDFExtractionError(ValueError):
    """Raised when PDF cannot be parsed or yields no text."""


def extract_pages(content: bytes) -> list[dict]:
    """
    Reads PDF bytes, returns list of {page: int, text: str} per page.
    Raises PDFExtractionError if no text is found across all pages.
    """
    doc = fitz.open(stream=content, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": page_num, "text": text.strip()})
    if not pages:
        raise PDFExtractionError(
            "No extractable text found. PDF may be image-only (scanned). "
            "OCR is not supported yet."
        )
    return pages
