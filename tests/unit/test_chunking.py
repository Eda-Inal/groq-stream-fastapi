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


def test_chunks_are_ordered() -> None:
    para = "Sentence one. Sentence two. Sentence three.\n\n"
    text = para * 40
    chunks = chunk_document(text, chunk_size_tokens=80, overlap_tokens=10, short_doc_max_tokens=0)
    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        assert c.chunk_index == i, f"Expected index {i}, got {c.chunk_index}"


def test_overlap_content_carried_over() -> None:
    # Each sentence ~10 tokens, chunk_size=40 → ~4 sentences per chunk
    # overlap=20 → next chunk should repeat last ~2 sentences of the previous
    sentences = [f"This is sentence number {i} in the document." for i in range(30)]
    text = " ".join(sentences)
    chunks = chunk_document(text, chunk_size_tokens=40, overlap_tokens=20, short_doc_max_tokens=0)
    assert len(chunks) >= 2
    for i in range(len(chunks) - 1):
        tail_words = set(chunks[i].text.split()[-30:])
        head_words = set(chunks[i + 1].text.split()[:30])
        shared = tail_words & head_words
        assert len(shared) >= 3, (
            f"No or insufficient overlap between chunk {i} and {i+1}: "
            f"shared_words={len(shared)}\n"
            f"  Chunk {i} tail: ...{' '.join(chunks[i].text.split()[-10:])}\n"
            f"  Chunk {i+1} head: {' '.join(chunks[i+1].text.split()[:10])}..."
        )


def test_no_content_loss() -> None:
    # All original tokens must be present when all chunk texts are joined
    sentences = [f"Unique token alpha{i} beta{i} gamma{i}." for i in range(50)]
    text = " ".join(sentences)
    chunks = chunk_document(text, chunk_size_tokens=60, overlap_tokens=15, short_doc_max_tokens=0)
    combined = " ".join(c.text for c in chunks)
    for i in range(50):
        assert f"alpha{i}" in combined, f"'alpha{i}' was lost during chunking"


def test_edge_case_empty_string() -> None:
    assert chunk_document("") == []


def test_edge_case_only_whitespace() -> None:
    assert chunk_document("   \n\n\t  ") == []


def test_edge_case_single_word() -> None:
    chunks = chunk_document("Hello")
    assert len(chunks) == 1
    assert chunks[0].text == "Hello"


def test_edge_case_repeated_newlines() -> None:
    text = "First paragraph.\n\n\n\n\n\nSecond paragraph."
    chunks = chunk_document(text, chunk_size_tokens=500, short_doc_max_tokens=0)
    joined = " ".join(c.text for c in chunks)
    assert "First paragraph" in joined
    assert "Second paragraph" in joined


def test_heading_starts_new_chunk() -> None:
    intro_body = "This section covers the basics of the topic. " * 8
    conclusion_body = "This section wraps up all findings and results. " * 8
    text = "# Introduction\n\n" + intro_body + "\n\n# Conclusion\n\n" + conclusion_body
    chunks = chunk_document(text, chunk_size_tokens=500, short_doc_max_tokens=0)
    joined_texts = [c.text for c in chunks]
    has_intro = any("Introduction" in t for t in joined_texts)
    has_conclusion = any("Conclusion" in t for t in joined_texts)
    assert has_intro, "Introduction heading not found in any chunk"
    assert has_conclusion, "Conclusion heading not found in any chunk"
    intro_idx = next(i for i, t in enumerate(joined_texts) if "Introduction" in t)
    conclusion_idx = next(i for i, t in enumerate(joined_texts) if "Conclusion" in t)
    assert intro_idx != conclusion_idx, "Introduction and Conclusion landed in the same chunk"


def test_heading_carried_in_chunk_text() -> None:
    body = "We collect your data to improve the service and personalize your experience. " * 5
    text = "# Privacy Policy\n\n" + body
    chunks = chunk_document(text, chunk_size_tokens=500, short_doc_max_tokens=0)
    assert any("Privacy Policy" in c.text for c in chunks), "Heading not found in any chunk"
    assert any("We collect" in c.text for c in chunks), "Body content not found in any chunk"


def test_multiple_heading_levels() -> None:
    body = "This paragraph contains detailed information about the topic at hand. " * 6
    text = (
        "# Chapter One\n\n" + body + "\n\n"
        "## Section 1.1\n\n" + body + "\n\n"
        "## Section 1.2\n\n" + body + "\n\n"
        "# Chapter Two\n\n" + body
    )
    chunks = chunk_document(text, chunk_size_tokens=500, short_doc_max_tokens=0)
    all_text = " ".join(c.text for c in chunks)
    for heading in ["Chapter One", "Section 1.1", "Section 1.2", "Chapter Two"]:
        assert heading in all_text, f"Heading '{heading}' was lost during chunking"


def test_heading_inside_code_fence_not_split() -> None:
    text = "Intro text.\n\n```python\n# this is a comment\nx = 1\n```\n\nOutro text."
    chunks = chunk_document(text, chunk_size_tokens=500, short_doc_max_tokens=0)
    joined = "\n".join(c.text for c in chunks)
    assert "# this is a comment" in joined, "Comment line inside code fence was lost"


def test_no_heading_text_unchanged() -> None:
    text = "Paragraph one. " * 20 + "\n\nParagraph two. " * 20
    chunks_before = chunk_document(text, chunk_size_tokens=200, overlap_tokens=20, short_doc_max_tokens=0)
    combined = " ".join(c.text for c in chunks_before)
    assert "Paragraph one" in combined
    assert "Paragraph two" in combined
