from app.services.rag_metrics import RagMetrics


def test_rag_metrics_snapshot_defaults() -> None:
    m = RagMetrics()
    snap = m.snapshot()
    assert snap["retrieval_calls_total"] == 0
    assert snap["recent_similarity_avg"] == 0.0
    assert snap["embedding_api_error_rate"] == 0.0


def test_rag_metrics_records_success_and_failure() -> None:
    m = RagMetrics()
    m.record_embedding_failure()
    m.record_retrieval(
        top_similarity=0.9,
        embedding_latency_ms=120,
        pgvector_query_ms=30,
        returned_chunks=3,
    )
    snap = m.snapshot()
    assert snap["retrieval_calls_total"] == 2
    assert snap["recent_similarity_avg"] == 0.9
    assert snap["embedding_api_error_rate"] == 0.5
