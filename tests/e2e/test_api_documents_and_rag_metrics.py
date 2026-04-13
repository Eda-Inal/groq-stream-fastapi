from fastapi.testclient import TestClient

from app.main import app
from app.schemas.document import DocumentIngestResponse
from app.services.rag_metrics import rag_metrics


def test_api_documents_ingest_and_rag_metrics(monkeypatch) -> None:
    from app.api.v1.endpoints import documents as documents_ep
    from app.api.v1.endpoints import rag_metrics as rag_metrics_ep

    async def _fake_ingest(*args, **kwargs):
        return DocumentIngestResponse(
            document_id=123,
            chunks_created=3,
            chunks_skipped=0,
            tokens_processed=120,
            elapsed_ms=15,
            embedding_model="nomic-embed-text",
        )

    async def _fake_count_docs(_db):
        return 4

    async def _fake_count_chunks(_db):
        return 19

    async def _fake_get_db():
        class _DummySession:
            pass

        yield _DummySession()

    monkeypatch.setattr(documents_ep.ingestion_service, "ingest_document", _fake_ingest)
    monkeypatch.setattr(rag_metrics_ep, "count_documents", _fake_count_docs)
    monkeypatch.setattr(rag_metrics_ep, "count_document_chunks", _fake_count_chunks)
    app.dependency_overrides.clear()
    app.dependency_overrides[documents_ep.get_db] = _fake_get_db
    app.dependency_overrides[rag_metrics_ep.get_db] = _fake_get_db

    # Seed in-memory metrics collector to validate endpoint payload.
    rag_metrics.record_retrieval(
        top_similarity=0.88,
        embedding_latency_ms=40,
        pgvector_query_ms=12,
        returned_chunks=3,
    )

    client = TestClient(app)
    ingest = client.post(
        "/api/v1/documents",
        json={
            "text": "hello document",
            "filename": "doc.txt",
            "document_type": "text",
            "tags": [],
        },
    )
    assert ingest.status_code == 200
    assert ingest.json()["document_id"] == 123

    metrics = client.get("/api/v1/rag/metrics")
    assert metrics.status_code == 200
    body = metrics.json()
    assert body["total_documents"] == 4
    assert body["total_chunks"] == 19
    assert "recent_similarity_avg" in body
