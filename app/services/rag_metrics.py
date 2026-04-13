from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RagSearchSample:
    similarity: float
    embedding_latency_ms: int
    pgvector_query_ms: int
    returned_chunks: int


class RagMetrics:
    """
    In-memory metrics for RAG retrieval health.
    """

    def __init__(self) -> None:
        self._recent_similarities: deque[float] = deque(maxlen=100)
        self._total_retrieval_calls = 0
        self._embedding_failures = 0
        self._recent_samples: deque[RagSearchSample] = deque(maxlen=100)

    def record_embedding_failure(self) -> None:
        self._total_retrieval_calls += 1
        self._embedding_failures += 1

    def record_retrieval(
        self,
        *,
        top_similarity: float,
        embedding_latency_ms: int,
        pgvector_query_ms: int,
        returned_chunks: int,
    ) -> None:
        self._total_retrieval_calls += 1
        self._recent_similarities.append(top_similarity)
        self._recent_samples.append(
            RagSearchSample(
                similarity=top_similarity,
                embedding_latency_ms=embedding_latency_ms,
                pgvector_query_ms=pgvector_query_ms,
                returned_chunks=returned_chunks,
            )
        )

    def snapshot(self) -> dict[str, float | int]:
        avg_similarity = (
            sum(self._recent_similarities) / len(self._recent_similarities)
            if self._recent_similarities
            else 0.0
        )
        avg_embedding_latency = (
            sum(s.embedding_latency_ms for s in self._recent_samples) / len(self._recent_samples)
            if self._recent_samples
            else 0.0
        )
        avg_pgvector_latency = (
            sum(s.pgvector_query_ms for s in self._recent_samples) / len(self._recent_samples)
            if self._recent_samples
            else 0.0
        )
        avg_returned_chunks = (
            sum(s.returned_chunks for s in self._recent_samples) / len(self._recent_samples)
            if self._recent_samples
            else 0.0
        )
        error_rate = (
            self._embedding_failures / self._total_retrieval_calls
            if self._total_retrieval_calls
            else 0.0
        )
        return {
            "recent_similarity_avg": round(avg_similarity, 4),
            "embedding_api_error_rate": round(error_rate, 4),
            "retrieval_calls_total": self._total_retrieval_calls,
            "recent_embedding_latency_ms_avg": round(avg_embedding_latency, 2),
            "recent_pgvector_latency_ms_avg": round(avg_pgvector_latency, 2),
            "recent_returned_chunks_avg": round(avg_returned_chunks, 2),
        }


rag_metrics = RagMetrics()
