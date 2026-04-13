import asyncio

import httpx

from app.core.config import settings
from app.services.embeddings import EmbeddingService


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, post_callable, **kwargs) -> None:
        self._post_callable = post_callable

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, path: str, json: dict):
        return await self._post_callable(path, json)


def test_embeddings_integration_http_error_returns_none(monkeypatch) -> None:
    async def _post_error(path: str, payload: dict):
        return _FakeResponse(503, {"error": "service unavailable"})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeAsyncClient(_post_error, **kwargs))

    service = EmbeddingService()
    result = asyncio.run(service.embed_text("integration http error"))
    assert result is None


def test_embeddings_integration_retry_exhaustion_returns_none(monkeypatch) -> None:
    calls = {"count": 0}

    async def _post_timeout(path: str, payload: dict):
        calls["count"] += 1
        raise httpx.TimeoutException("timeout")

    async def _fake_sleep(_seconds: float):
        return None

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeAsyncClient(_post_timeout, **kwargs))
    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(settings, "embedding_max_retries", 3)

    service = EmbeddingService()
    result = asyncio.run(service.embed_text("integration timeout"))
    assert result is None
    assert calls["count"] == 3
