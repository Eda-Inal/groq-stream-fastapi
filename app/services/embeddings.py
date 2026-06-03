from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from app.core.config import settings

logger = structlog.get_logger()


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    model_name: str


class EmbeddingService:
    """
    Embedding client with retry/backoff and optional in-memory cache.

    - Returns None on all failures (never raises to caller).
    - Retries on timeout/network transport issues.
    - Logs latency and failures for observability.
    """

    def __init__(self) -> None:
        self._cache: dict[str, EmbeddingResult] = {}
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            api_key = settings.embedding_api_key or settings.groq_api_key
            base_url = settings.embedding_base_url or settings.groq_base_url
            self._client = httpx.AsyncClient(
                base_url=base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(settings.embedding_timeout),
                verify=getattr(settings, "groq_verify_ssl", True),
            )
        return self._client

    def _cache_key(self, text: str, model_name: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{model_name}:{digest}"

    def _cache_get(self, key: str) -> EmbeddingResult | None:
        if not settings.embedding_cache_enabled:
            return None
        return self._cache.get(key)

    def _cache_set(self, key: str, value: EmbeddingResult) -> None:
        if not settings.embedding_cache_enabled:
            return
        if len(self._cache) >= settings.embedding_cache_max_entries:
            # Keep cache simple and deterministic; remove oldest inserted key.
            oldest = next(iter(self._cache), None)
            if oldest is not None:
                self._cache.pop(oldest, None)
        self._cache[key] = value

    async def embed_batch(
        self, texts: list[str], model_name: str | None = None
    ) -> list[EmbeddingResult] | None:
        """
        Embed multiple texts in a single API call.

        Returns a list of EmbeddingResult in the same order as input.
        Returns None if any single embedding fails — caller must treat this as
        a hard failure and not persist partial results.
        """
        if not texts:
            return []

        target_model = model_name or settings.embedding_model_name
        results: list[EmbeddingResult | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                logger.error("embedding_batch_invalid_input", index=i)
                return None
            cache_key = self._cache_key(text, target_model)
            cached = self._cache_get(cache_key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            return results  # type: ignore[return-value]

        # Split into mini-batches so a single large PDF never sends tens of
        # thousands of tokens in one HTTP request, which would hit API payload
        # limits and timeouts.
        batch_size = max(1, settings.embedding_batch_size)
        mini_batches = [
            (uncached_indices[i : i + batch_size], uncached_texts[i : i + batch_size])
            for i in range(0, len(uncached_texts), batch_size)
        ]

        log = logger.bind(
            model=target_model,
            total_texts=len(uncached_texts),
            mini_batches=len(mini_batches),
            batch_size=batch_size,
        )

        for mb_num, (mb_indices, mb_texts) in enumerate(mini_batches):
            mini_result = await self._embed_mini_batch(
                texts=texts,
                mb_texts=mb_texts,
                mb_indices=mb_indices,
                target_model=target_model,
                results=results,
            )
            if mini_result is None:
                log.error("embedding_batch_mini_batch_failed", mini_batch=mb_num)
                return None

        log.info("embedding_batch_success", total_texts=len(uncached_texts))
        return results  # type: ignore[return-value]

    async def _embed_mini_batch(
        self,
        *,
        texts: list[str],
        mb_texts: list[str],
        mb_indices: list[int],
        target_model: str,
        results: list,
    ) -> list | None:
        """Send one mini-batch to the API with retry. Fills results in-place.
        Returns results on success, None on unrecoverable failure."""
        payload: dict[str, Any] = {"model": target_model, "input": mb_texts}
        retries = max(1, settings.embedding_max_retries)
        log = logger.bind(model=target_model, batch_size=len(mb_texts))

        for attempt in range(retries):
            started = time.perf_counter()
            try:
                client = await self._get_client()
                response = await client.post("/embeddings", json=payload)
                latency_ms = int((time.perf_counter() - started) * 1000)

                if response.status_code >= 400:
                    log.error("embedding_batch_http_error",
                              status=response.status_code, latency_ms=latency_ms)
                    return None

                data = response.json()
                rows = data.get("data") if isinstance(data, dict) else None
                if not isinstance(rows, list) or len(rows) != len(mb_texts):
                    log.error("embedding_batch_invalid_shape", latency_ms=latency_ms)
                    return None

                rows_sorted = sorted(rows, key=lambda r: r.get("index", 0) if isinstance(r, dict) else 0)
                resolved_model = (data.get("model") if isinstance(data, dict) else None) or target_model

                for batch_i, row in enumerate(rows_sorted):
                    vector = row.get("embedding") if isinstance(row, dict) else None
                    if not isinstance(vector, list):
                        log.error("embedding_batch_missing_vector", latency_ms=latency_ms)
                        return None
                    try:
                        normalized = [float(v) for v in vector]
                    except (TypeError, ValueError):
                        log.error("embedding_batch_non_numeric", latency_ms=latency_ms)
                        return None
                    if len(normalized) != settings.embedding_dim:
                        log.error("embedding_batch_dimension_mismatch",
                                  expected=settings.embedding_dim, got=len(normalized))
                        return None

                    orig_idx = mb_indices[batch_i]
                    emb_result = EmbeddingResult(vector=normalized, model_name=resolved_model)
                    results[orig_idx] = emb_result
                    self._cache_set(self._cache_key(texts[orig_idx], target_model), emb_result)

                log.info("embedding_mini_batch_success", latency_ms=latency_ms)
                return results

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                if attempt < retries - 1:
                    wait_s = settings.embedding_retry_backoff ** attempt
                    log.warning("embedding_batch_retryable_error",
                                attempt=attempt + 1, wait_seconds=wait_s, error=str(exc))
                    await asyncio.sleep(wait_s)
                    continue
                log.error("embedding_batch_all_retries_exhausted", attempts=retries, error=str(exc))
                return None
            except Exception as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                log.error("embedding_batch_unexpected_error", error=str(exc),
                          latency_ms=latency_ms, exc_info=True)
                return None

        return None

    async def embed_text(self, text: str, model_name: str | None = None) -> EmbeddingResult | None:
        if not isinstance(text, str) or not text.strip():
            return None

        target_model = model_name or settings.embedding_model_name
        cache_key = self._cache_key(text, target_model)
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("embedding_cache_hit", model=target_model)
            return cached

        payload: dict[str, Any] = {"model": target_model, "input": text}
        retries = max(1, settings.embedding_max_retries)

        log = logger.bind(model=target_model)

        for attempt in range(retries):
            started = time.perf_counter()
            try:
                client = await self._get_client()
                response = await client.post("/embeddings", json=payload)

                latency_ms = int((time.perf_counter() - started) * 1000)
                if response.status_code >= 400:
                    log.error(
                        "embedding_api_http_error",
                        status=response.status_code,
                        embedding_latency_ms=latency_ms,
                    )
                    return None

                data = response.json()
                rows = data.get("data") if isinstance(data, dict) else None
                if not isinstance(rows, list) or not rows:
                    log.error("embedding_api_invalid_shape", embedding_latency_ms=latency_ms)
                    return None

                row0 = rows[0] if isinstance(rows[0], dict) else None
                vector = row0.get("embedding") if isinstance(row0, dict) else None
                if not isinstance(vector, list):
                    log.error("embedding_missing_vector", embedding_latency_ms=latency_ms)
                    return None

                try:
                    normalized_vector = [float(v) for v in vector]
                except (TypeError, ValueError):
                    log.error("embedding_non_numeric_vector", embedding_latency_ms=latency_ms)
                    return None

                if len(normalized_vector) != settings.embedding_dim:
                    log.error(
                        "embedding_dimension_mismatch",
                        expected=settings.embedding_dim,
                        got=len(normalized_vector),
                        embedding_latency_ms=latency_ms,
                    )
                    return None

                resolved_model = data.get("model") if isinstance(data, dict) else None
                result = EmbeddingResult(
                    vector=normalized_vector,
                    model_name=resolved_model or target_model,
                )
                self._cache_set(cache_key, result)
                log.info("embedding_success", embedding_latency_ms=latency_ms)
                return result

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                if attempt < retries - 1:
                    wait_s = settings.embedding_retry_backoff ** attempt
                    log.warning(
                        "embedding_retryable_error",
                        attempt=attempt + 1,
                        max_attempts=retries,
                        wait_seconds=wait_s,
                        error=str(exc),
                        embedding_latency_ms=latency_ms,
                    )
                    await asyncio.sleep(wait_s)
                    continue

                log.error(
                    "embedding_all_retries_exhausted",
                    attempts=retries,
                    error=str(exc),
                    embedding_latency_ms=latency_ms,
                )
                return None
            except Exception as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                log.error(
                    "embedding_unexpected_error",
                    error=str(exc),
                    embedding_latency_ms=latency_ms,
                    exc_info=True,
                )
                return None

        log.error("embedding_unreachable_state")
        return None
