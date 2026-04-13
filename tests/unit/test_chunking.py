import pytest

from app.core.config import settings
from app.services.chunking import (
    ChunkRecord,
    DocumentTooLargeError,
    chunk_document,
    count_tokens,
    normalize_document_text,
)


def test_normalize_strips_nuls_and_unifies_newlines() -> None:
    raw = "a\x00b\r\nc\n\n\n\nd"
    out = normalize_document_text(raw)
    assert "\x00" not in out
    assert "\r" not in out
    assert "\n\n" in out


def test_count_tokens_empty() -> None:
    assert count_tokens("") == 0


def test_short_doc_single_chunk() -> None:
    text = "Hello world. " * 5
    assert count_tokens(text) <= settings.short_doc_single_chunk_max_tokens
    chunks = chunk_document(text)
    assert len(chunks) == 1
    assert isinstance(chunks[0], ChunkRecord)
    assert chunks[0].text.strip() == text.strip()


def test_document_too_large_raises() -> None:
    # Limit must compare to normalized text token count (same as chunk_document)
    filler = "word " * 200
    text = filler * 50
    normalized = normalize_document_text(text)
    with pytest.raises(DocumentTooLargeError):
        chunk_document(
            text,
            max_document_tokens=count_tokens(normalized) - 1,
        )


def test_chunk_respects_max_size_roughly() -> None:
    # Long repeated paragraph so we get multiple chunks
    para = "Sentence one. Sentence two. Sentence three.\n\n"
    text = para * 80
    chunks = chunk_document(text, chunk_size_tokens=80, overlap_tokens=10, short_doc_max_tokens=0)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= 100  # small slack above 80 for boundary effects


def test_code_fence_kept_when_small() -> None:
    text = 'Intro.\n\n```python\nx = 1\n```\n\nOutro.'
    chunks = chunk_document(text, chunk_size_tokens=500, short_doc_max_tokens=0)
    joined = "\n".join(c.text for c in chunks)
    assert "```python" in joined
    assert "```" in joined


def test_metadata_passed_through() -> None:
    chunks = chunk_document(
        "a " * 30,
        short_doc_max_tokens=0,
        page_number=3,
        section_heading="Policy",
    )
    assert chunks and all(c.page_number == 3 for c in chunks)
    assert chunks[0].section_heading == "Policy"


def test_min_chunk_drops_only_when_multiple() -> None:
    # Force many tiny pieces then rely on MIN_CHUNK_TOKENS drop
    text = "x " * 500
    chunks = chunk_document(
        text,
        chunk_size_tokens=15,
        overlap_tokens=0,
        short_doc_max_tokens=0,
        min_chunk_tokens=10,
    )
    assert chunks
    assert all(c.token_count >= 10 for c in chunks)
