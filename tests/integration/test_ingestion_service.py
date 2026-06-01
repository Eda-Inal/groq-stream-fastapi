import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

# fitz (pymupdf) may not be installed in the CI/test environment.
# Mock it at module level so pdf_extractor.py can be imported during collection.
if 'fitz' not in sys.modules:
    sys.modules['fitz'] = MagicMock()

from app.schemas.document import DocumentIngestRequest
from app.services.chunking import ChunkRecord
from app.services.embeddings import EmbeddingResult
from app.services.ingestion_service import IngestionService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeSession:
    def __init__(self) -> None:
        self.committed = False
        self.refreshed = False
        self.flushed = False

    async def commit(self) -> None:
        self.committed = True

    async def refresh(self, _obj) -> None:
        self.refreshed = True

    async def flush(self) -> None:
        self.flushed = True


def _make_chunk(text: str, token_count: int, page_number: int | None = None) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=str(uuid4()),
        doc_id=0,
        chunk_index=0,
        total_chunks=0,
        text=text,
        token_count=token_count,
        source_filename="",
        page_number=page_number,
        section_heading=None,
        context_prefix="",
    )


def _fake_embeddings(texts: list[str]) -> list[EmbeddingResult]:
    return [EmbeddingResult(vector=[0.1] * 768, model_name="nomic-embed-text") for _ in texts]


# ---------------------------------------------------------------------------
# ingest_document
# ---------------------------------------------------------------------------

def test_ingest_document_happy_path(monkeypatch) -> None:
    service = IngestionService()
    session = _FakeSession()
    created_chunks: list[dict] = []
    fake_doc = SimpleNamespace(id=42, chunk_count=0, embedding_model_name=None)

    monkeypatch.setattr(
        "app.services.ingestion_service.chunk_document",
        lambda *args, **kwargs: [
            _make_chunk("chunk one", 10),
            _make_chunk("chunk two", 12),
        ],
    )

    async def _fake_get_doc(*a, **kw): return None
    async def _fake_create_document(*args, **kwargs): return fake_doc
    async def _fake_create_chunk(*args, **kwargs): created_chunks.append(kwargs)
    async def _fake_embed_batch(texts, **kwargs): return _fake_embeddings(texts)

    monkeypatch.setattr("app.services.ingestion_service.get_document_by_filename", _fake_get_doc)
    monkeypatch.setattr("app.services.ingestion_service.create_document", _fake_create_document)
    monkeypatch.setattr("app.services.ingestion_service.create_document_chunk", _fake_create_chunk)
    monkeypatch.setattr(service.embeddings, "embed_batch", _fake_embed_batch)

    payload = DocumentIngestRequest(
        text="sample doc text",
        filename="doc.txt",
        source="unit-test",
        document_type="text",
        tags=["test"],
        user_id="u1",
    )

    result = asyncio.run(service.ingest_document(session=session, payload=payload))

    assert result.document_id == 42
    assert result.chunks_created == 2
    assert result.tokens_processed == 22
    assert session.committed is True
    assert session.refreshed is True
    assert len(created_chunks) == 2


def test_ingest_document_embed_batch_fails_raises(monkeypatch) -> None:
    service = IngestionService()
    session = _FakeSession()
    fake_doc = SimpleNamespace(id=11, chunk_count=0, embedding_model_name=None)

    monkeypatch.setattr(
        "app.services.ingestion_service.chunk_document",
        lambda *args, **kwargs: [_make_chunk("a", 1)],
    )

    async def _fake_get_doc(*a, **kw): return None
    async def _fake_create_document(*args, **kwargs): return fake_doc
    async def _embed_none(texts, **kwargs): return None

    monkeypatch.setattr("app.services.ingestion_service.get_document_by_filename", _fake_get_doc)
    monkeypatch.setattr("app.services.ingestion_service.create_document", _fake_create_document)
    monkeypatch.setattr(service.embeddings, "embed_batch", _embed_none)

    payload = DocumentIngestRequest(
        text="short",
        filename="doc.txt",
        document_type="text",
        tags=[],
    )

    with pytest.raises(RuntimeError):
        asyncio.run(service.ingest_document(session=session, payload=payload))
    assert session.committed is False


# ---------------------------------------------------------------------------
# ingest_pdf — cross-page overlap
# ---------------------------------------------------------------------------

def test_ingest_pdf_overlap_spans_page_boundary(monkeypatch) -> None:
    """Chunks should be able to span the page 1→2 boundary after the merge fix."""
    import app.services.pdf_extractor as _pe

    service = IngestionService()
    session = _FakeSession()
    created_chunks: list[dict] = []
    fake_doc = SimpleNamespace(id=7, chunk_count=0, embedding_model_name=None)

    # Page 1 ends mid-sentence; page 2 finishes it.
    page1 = "Regular content on page one. " * 4 + "This sentence starts on page one"
    page2 = " and finishes on page two. " + "More content on page two. " * 4

    monkeypatch.setattr(_pe, "extract_pages", lambda _: [
        {"page": 1, "text": page1},
        {"page": 2, "text": page2},
    ])

    async def _fake_get_doc(*a, **kw): return None
    async def _fake_create_doc(*a, **kw): return fake_doc
    async def _fake_create_chunk(*a, **kw): created_chunks.append(kw)
    async def _fake_embed_batch(texts, **kw): return _fake_embeddings(texts)

    monkeypatch.setattr("app.services.ingestion_service.get_document_by_filename", _fake_get_doc)
    monkeypatch.setattr("app.services.ingestion_service.create_document", _fake_create_doc)
    monkeypatch.setattr("app.services.ingestion_service.create_document_chunk", _fake_create_chunk)
    monkeypatch.setattr(service.embeddings, "embed_batch", _fake_embed_batch)

    result = asyncio.run(service.ingest_pdf(
        session=session, content=b"fake", filename="test.pdf", user_id=None, tags=[],
    ))

    assert result.chunks_created > 0
    all_text = " ".join(kw["text"] for kw in created_chunks)
    assert "starts on page one" in all_text
    assert "finishes on page two" in all_text

    # Key assertion: at least one chunk contains text from both pages.
    cross_chunk = next(
        (kw for kw in created_chunks
         if "starts on page one" in kw["text"] and "finishes on page two" in kw["text"]),
        None,
    )
    assert cross_chunk is not None, "No chunk spans the page boundary — overlap fix not working"


def test_ingest_pdf_page_numbers_assigned(monkeypatch) -> None:
    """Each chunk should carry the page_number of the page its content came from."""
    import app.services.pdf_extractor as _pe

    service = IngestionService()
    session = _FakeSession()
    created_chunks: list[dict] = []
    fake_doc = SimpleNamespace(id=8, chunk_count=0, embedding_model_name=None)

    # Two clearly distinct pages — long enough (~350 tokens each) so each page
    # produces at least one chunk that doesn't overlap the page boundary.
    # Non-sequential page numbers verify the mapping isn't positional.
    page1 = "Alpha unique content page one value. " * 50   # ~350 tokens
    page3 = "Beta unique content page three value. " * 50  # ~350 tokens

    monkeypatch.setattr(_pe, "extract_pages", lambda _: [
        {"page": 1, "text": page1},
        {"page": 3, "text": page3},
    ])

    async def _fake_get_doc(*a, **kw): return None
    async def _fake_create_doc(*a, **kw): return fake_doc
    async def _fake_create_chunk(*a, **kw): created_chunks.append(kw)
    async def _fake_embed_batch(texts, **kw): return _fake_embeddings(texts)

    monkeypatch.setattr("app.services.ingestion_service.get_document_by_filename", _fake_get_doc)
    monkeypatch.setattr("app.services.ingestion_service.create_document", _fake_create_doc)
    monkeypatch.setattr("app.services.ingestion_service.create_document_chunk", _fake_create_chunk)
    monkeypatch.setattr(service.embeddings, "embed_batch", _fake_embed_batch)

    asyncio.run(service.ingest_pdf(
        session=session, content=b"fake", filename="test.pdf", user_id=None, tags=[],
    ))

    # Boundary chunks may contain text from both pages — exclude them.
    # Only check chunks that exclusively contain content from one page.
    pure_p1 = [
        kw for kw in created_chunks
        if "Alpha" in kw.get("text", "") and "Beta" not in kw.get("text", "")
    ]
    pure_p3 = [
        kw for kw in created_chunks
        if "Beta" in kw.get("text", "") and "Alpha" not in kw.get("text", "")
    ]

    assert pure_p1, "No pure page-1 chunks found"
    assert pure_p3, "No pure page-3 chunks found"
    assert all(kw.get("page_number") == 1 for kw in pure_p1), \
        f"Wrong page_number for page-1 chunks: {[kw.get('page_number') for kw in pure_p1]}"
    assert all(kw.get("page_number") == 3 for kw in pure_p3), \
        f"Wrong page_number for page-3 chunks: {[kw.get('page_number') for kw in pure_p3]}"


def test_ingest_pdf_single_page_unchanged(monkeypatch) -> None:
    """Single-page PDF should behave the same as before."""
    import app.services.pdf_extractor as _pe

    service = IngestionService()
    session = _FakeSession()
    created_chunks: list[dict] = []
    fake_doc = SimpleNamespace(id=9, chunk_count=0, embedding_model_name=None)

    page_text = "Single page content. " * 20

    monkeypatch.setattr(_pe, "extract_pages", lambda _: [{"page": 1, "text": page_text}])

    async def _fake_get_doc(*a, **kw): return None
    async def _fake_create_doc(*a, **kw): return fake_doc
    async def _fake_create_chunk(*a, **kw): created_chunks.append(kw)
    async def _fake_embed_batch(texts, **kw): return _fake_embeddings(texts)

    monkeypatch.setattr("app.services.ingestion_service.get_document_by_filename", _fake_get_doc)
    monkeypatch.setattr("app.services.ingestion_service.create_document", _fake_create_doc)
    monkeypatch.setattr("app.services.ingestion_service.create_document_chunk", _fake_create_chunk)
    monkeypatch.setattr(service.embeddings, "embed_batch", _fake_embed_batch)

    result = asyncio.run(service.ingest_pdf(
        session=session, content=b"fake", filename="single.pdf", user_id=None, tags=[],
    ))

    assert result.chunks_created > 0
    assert all(kw.get("page_number") == 1 for kw in created_chunks)
    assert session.committed is True
