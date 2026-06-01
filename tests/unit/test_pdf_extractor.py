"""
Unit tests for pdf_extractor.extract_pages() and its interaction with
chunk_document().

fitz (pymupdf) is mocked at module level — no real PDF is needed.
"""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

if "fitz" not in sys.modules:
    sys.modules["fitz"] = MagicMock()

import app.services.pdf_extractor as _pe
from app.services.pdf_extractor import PDFExtractionError, extract_pages


# ---------------------------------------------------------------------------
# Fake fitz helpers
# ---------------------------------------------------------------------------

def _make_page(plain_text: str = "", blocks: list | None = None) -> object:
    """
    Fake fitz page.
    get_text("dict") returns {"blocks": blocks} — empty blocks trigger the
    fallback path in _page_to_text_with_headings() which calls get_text("text").
    get_text("text") returns plain_text.
    """
    def get_text(mode: str):
        if mode == "dict":
            return {"blocks": blocks or []}
        return plain_text
    return SimpleNamespace(get_text=get_text)


def _make_doc(*pages) -> object:
    """Fake fitz document that iterates over the given page objects."""
    class _Doc:
        def __iter__(self): return iter(pages)
    return _Doc()


# ---------------------------------------------------------------------------
# extract_pages — basic behaviour
# (empty blocks → _page_to_text_with_headings falls back to get_text("text"))
# ---------------------------------------------------------------------------

def test_text_passed_through(monkeypatch) -> None:
    page = _make_page(plain_text="Some document content here.")
    monkeypatch.setattr(_pe.fitz, "open", lambda stream, filetype: _make_doc(page))
    result = extract_pages(b"fake")
    assert len(result) == 1
    assert result[0]["page"] == 1
    assert "Some document content here." in result[0]["text"]


def test_page_number_assigned_correctly(monkeypatch) -> None:
    pages = [_make_page(plain_text=f"Page {i} content.") for i in range(1, 4)]
    monkeypatch.setattr(_pe.fitz, "open", lambda stream, filetype: _make_doc(*pages))
    result = extract_pages(b"fake")
    assert [r["page"] for r in result] == [1, 2, 3]


def test_empty_blocks_falls_back_to_plain_text(monkeypatch) -> None:
    # blocks=[] → no font sizes → fallback to get_text("text")
    page = _make_page(plain_text="Plain fallback text.")
    monkeypatch.setattr(_pe.fitz, "open", lambda stream, filetype: _make_doc(page))
    result = extract_pages(b"fake")
    assert len(result) == 1
    assert "Plain fallback text." in result[0]["text"]


def test_whitespace_only_plain_text_skips_page(monkeypatch) -> None:
    page = _make_page(plain_text="   \n\n  ")
    monkeypatch.setattr(_pe.fitz, "open", lambda stream, filetype: _make_doc(page))
    import pytest
    with pytest.raises(PDFExtractionError):
        extract_pages(b"fake")


def test_empty_page_skipped(monkeypatch) -> None:
    p1 = _make_page(plain_text="Real content.")
    p2 = _make_page(plain_text="")        # empty — skipped
    p3 = _make_page(plain_text="More content.")
    monkeypatch.setattr(_pe.fitz, "open", lambda stream, filetype: _make_doc(p1, p2, p3))
    result = extract_pages(b"fake")
    assert len(result) == 2
    assert result[0]["page"] == 1
    assert result[1]["page"] == 3


def test_no_text_at_all_raises(monkeypatch) -> None:
    p = _make_page(plain_text="")
    monkeypatch.setattr(_pe.fitz, "open", lambda stream, filetype: _make_doc(p))
    import pytest
    with pytest.raises(PDFExtractionError):
        extract_pages(b"fake")


# ---------------------------------------------------------------------------
# Integration: PyMuPDF markdown output → chunk_document → section_heading
# ---------------------------------------------------------------------------

def test_markdown_heading_becomes_section_heading() -> None:
    """Heading detected by PyMuPDF propagates to ChunkRecord.section_heading."""
    from app.services.chunking import chunk_document

    md = "# Privacy Policy\n\n" + "We collect your data to improve the service. " * 6
    chunks = chunk_document(md, chunk_size_tokens=500, short_doc_max_tokens=0)
    assert chunks
    assert all(c.section_heading == "Privacy Policy" for c in chunks)


def test_bold_heading_markers_stripped_from_section_heading() -> None:
    """PyMuPDF may emit '# **Title**'; section_heading should store 'Title'."""
    from app.services.chunking import chunk_document

    md = "# **Bold Section**\n\n" + "Section content goes here. " * 6
    chunks = chunk_document(md, chunk_size_tokens=500, short_doc_max_tokens=0)
    assert chunks
    assert all(c.section_heading == "Bold Section" for c in chunks), (
        f"Expected 'Bold Section', got {[c.section_heading for c in chunks]}"
    )


def test_markdown_table_preserved_in_chunk_text() -> None:
    """Tables from PyMuPDF markdown mode stay intact in chunk text."""
    from app.services.chunking import chunk_document

    md = (
        "# Price List\n\n"
        "| Item   | Price |\n"
        "|--------|-------|\n"
        "| Table  | 500   |\n"
        "| Chair  | 200   |\n"
    )
    chunks = chunk_document(md, chunk_size_tokens=500, short_doc_max_tokens=0)
    joined = "\n".join(c.text for c in chunks)
    assert "| Item" in joined
    assert "| Table" in joined


def test_multiple_headings_each_gets_own_section_heading() -> None:
    from app.services.chunking import chunk_document

    body = "Detailed content about this topic. " * 6
    md = "# Chapter One\n\n" + body + "\n\n# Chapter Two\n\n" + body
    chunks = chunk_document(md, chunk_size_tokens=500, short_doc_max_tokens=0)
    ch1 = [c for c in chunks if "Chapter One" in c.text]
    ch2 = [c for c in chunks if "Chapter Two" in c.text]
    assert ch1 and all(c.section_heading == "Chapter One" for c in ch1)
    assert ch2 and all(c.section_heading == "Chapter Two" for c in ch2)
