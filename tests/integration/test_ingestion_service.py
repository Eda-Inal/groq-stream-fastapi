import asyncio
from types import SimpleNamespace

import pytest

from app.schemas.document import DocumentIngestRequest
from app.services.chunking import ChunkRecord
from app.services.embeddings import EmbeddingResult
from app.services.ingestion_service import IngestionService


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


def test_ingestion_service_partial_chunk_success(monkeypatch) -> None:
    service = IngestionService()
    session = _FakeSession()
    created_chunks: list[dict] = []
    fake_doc = SimpleNamespace(id=42, chunk_count=0, embedding_model_name=None)

    monkeypatch.setattr(
        "app.services.ingestion_service.chunk_document",
        lambda *args, **kwargs: [
            ChunkRecord(text="chunk one", token_count=10),
            ChunkRecord(text="chunk two", token_count=12),
        ],
    )

    async def _fake_create_document(*args, **kwargs):
        return fake_doc

    async def _fake_create_chunk(*args, **kwargs):
        created_chunks.append(kwargs)
        return SimpleNamespace(id=len(created_chunks))

    async def _fake_embed(text: str, model_name=None):
        if "two" in text:
            return None
        return EmbeddingResult(vector=[0.2] * 768, model_name="nomic-embed-text")

    monkeypatch.setattr("app.services.ingestion_service.create_document", _fake_create_document)
    monkeypatch.setattr("app.services.ingestion_service.create_document_chunk", _fake_create_chunk)
    monkeypatch.setattr(service.embeddings, "embed_text", _fake_embed)

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
    assert result.chunks_created == 1
    assert result.chunks_skipped == 1
    assert result.tokens_processed == 10
    assert session.committed is True
    assert session.refreshed is True
    assert fake_doc.chunk_count == 1
    assert fake_doc.embedding_model_name == "nomic-embed-text"
    assert len(created_chunks) == 1


def test_ingestion_service_all_embeddings_fail_raises(monkeypatch) -> None:
    service = IngestionService()
    session = _FakeSession()
    fake_doc = SimpleNamespace(id=11, chunk_count=0, embedding_model_name=None)

    monkeypatch.setattr(
        "app.services.ingestion_service.chunk_document",
        lambda *args, **kwargs: [ChunkRecord(text="a", token_count=1)],
    )

    async def _fake_create_document(*args, **kwargs):
        return fake_doc

    async def _embed_none(*args, **kwargs):
        return None

    monkeypatch.setattr("app.services.ingestion_service.create_document", _fake_create_document)
    monkeypatch.setattr(service.embeddings, "embed_text", _embed_none)

    payload = DocumentIngestRequest(
        text="short",
        filename="doc.txt",
        document_type="text",
        tags=[],
    )

    with pytest.raises(RuntimeError):
        asyncio.run(service.ingest_document(session=session, payload=payload))
    assert session.committed is False
