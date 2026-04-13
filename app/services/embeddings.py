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

    async def embed_text(self, text: str, model_name: str | None = None) -> EmbeddingResult | None:
        if not isinstance(text, str) or not text.strip():
            return None

        target_model = model_name or settings.embedding_model_name
        cache_key = self._cache_key(text, target_model)
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("embedding_cache_hit", model=target_model)
            return cached

        api_key = settings.embedding_api_key or settings.groq_api_key
        base_url = settings.embedding_base_url or settings.groq_base_url
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"model": target_model, "input": text}
        timeout = httpx.Timeout(settings.embedding_timeout)
        retries = max(1, settings.embedding_max_retries)

        log = logger.bind(model=target_model)

        for attempt in range(retries):
            started = time.perf_counter()
            try:
                async with httpx.AsyncClient(
                    base_url=base_url,
                    headers=headers,
                    timeout=timeout,
                    verify=getattr(settings, "groq_verify_ssl", True),
                ) as client:
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
