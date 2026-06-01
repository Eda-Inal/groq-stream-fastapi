from __future__ import annotations

from collections import Counter

import fitz  # pymupdf

# A span whose font size >= body_size * this ratio is a heading candidate.
_HEADING_SIZE_RATIO: float = 1.15
# Lines longer than this are never treated as headings (e.g. full-width table rows).
_MAX_HEADING_CHARS: int = 200


class PDFExtractionError(ValueError):
    """Raised when PDF cannot be parsed or yields no text."""


def _page_to_text_with_headings(page) -> str:
    """
    Reconstruct page text from get_text("dict") blocks.

    Lines whose maximum span font size exceeds body_size * _HEADING_SIZE_RATIO
    are prefixed with "# " so that chunk_document() recognises them as markdown
    headings and stores them in ChunkRecord.section_heading.

    Falls back to plain get_text("text") when no font-size data is available.
    """
    raw = page.get_text("dict")
    blocks = [b for b in raw.get("blocks", []) if b.get("type") == 0]

    sizes: list[float] = [
        round(span.get("size", 0.0), 1)
        for block in blocks
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span.get("text", "").strip() and span.get("size", 0.0) > 0
    ]

    if not sizes:
        return page.get_text("text")

    body_size: float = Counter(sizes).most_common(1)[0][0]
    heading_threshold: float = body_size * _HEADING_SIZE_RATIO

    output_blocks: list[str] = []
    for block in blocks:
        block_lines: list[str] = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(s.get("text", "") for s in spans)
            stripped = line_text.strip()
            if not stripped:
                continue

            max_size = max(
                (s.get("size", 0.0) for s in spans if s.get("text", "").strip()),
                default=0.0,
            )
            is_heading = (
                len(stripped) <= _MAX_HEADING_CHARS
                and max_size >= heading_threshold
            )
            block_lines.append(f"# {stripped}" if is_heading else line_text)

        if block_lines:
            output_blocks.append("\n".join(block_lines))

    return "\n\n".join(output_blocks)


def extract_pages(content: bytes) -> list[dict]:
    """
    Reads PDF bytes, returns list of {page: int, text: str} per page.
    text contains markdown-style "# heading" markers for lines detected as
    headings via font-size analysis, enabling section_heading propagation
    downstream in chunk_document().
    Raises PDFExtractionError if no text is found across all pages.
    """
    doc = fitz.open(stream=content, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = _page_to_text_with_headings(page)
        if text.strip():
            pages.append({"page": page_num, "text": text.strip()})
    if not pages:
        raise PDFExtractionError(
            "No extractable text found. PDF may be image-only (scanned). "
            "OCR is not supported yet."
        )
    return pages
