import asyncio

import httpx

from app.core.config import settings
from app.services.embeddings import EmbeddingResult, EmbeddingService


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


async def _ok_post(path: str, payload: dict) -> _FakeResponse:
    assert path == "/embeddings"
    return _FakeResponse(
        200,
        {
            "model": payload["model"],
            "data": [{"embedding": [0.1] * settings.embedding_dim}],
        },
    )


def test_embed_success_returns_vector_and_model(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeAsyncClient(_ok_post, **kwargs))
    svc = EmbeddingService()

    result = asyncio.run(svc.embed_text("hello embeddings"))

    assert isinstance(result, EmbeddingResult)
    assert result is not None
    assert result.model_name == settings.embedding_model_name
    assert len(result.vector) == settings.embedding_dim


def test_embed_retries_on_timeout_then_succeeds(monkeypatch) -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    async def _flaky_post(path: str, payload: dict):
        calls["count"] += 1
        if calls["count"] < 3:
            raise httpx.TimeoutException("timeout")
        return await _ok_post(path, payload)

    async def _fake_sleep(seconds: float):
        sleeps.append(seconds)
        return None

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeAsyncClient(_flaky_post, **kwargs))
    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(settings, "embedding_max_retries", 3)
    monkeypatch.setattr(settings, "embedding_retry_backoff", 2.0)

    svc = EmbeddingService()
    result = asyncio.run(svc.embed_text("retry test"))

    assert result is not None
    assert calls["count"] == 3
    assert sleeps == [1.0, 2.0]


def test_embed_uses_cache_on_repeat_calls(monkeypatch) -> None:
    calls = {"count": 0}

    async def _counting_post(path: str, payload: dict):
        calls["count"] += 1
        return await _ok_post(path, payload)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeAsyncClient(_counting_post, **kwargs))
    monkeypatch.setattr(settings, "embedding_cache_enabled", True)
    monkeypatch.setattr(settings, "embedding_cache_max_entries", 100)

    svc = EmbeddingService()
    first = asyncio.run(svc.embed_text("cached text"))
    second = asyncio.run(svc.embed_text("cached text"))

    assert first is not None and second is not None
    assert calls["count"] == 1


def test_embed_dimension_mismatch_returns_none(monkeypatch) -> None:
    async def _bad_dim_post(path: str, payload: dict):
        return _FakeResponse(
            200,
            {
                "model": payload["model"],
                "data": [{"embedding": [0.1] * 10}],
            },
        )

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeAsyncClient(_bad_dim_post, **kwargs))
    svc = EmbeddingService()

    result = asyncio.run(svc.embed_text("bad dim"))
    assert result is None
