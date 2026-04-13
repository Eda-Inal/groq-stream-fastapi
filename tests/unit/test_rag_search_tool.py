from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

from app.mcp_server.tools.rag_search import RagSearchTool
from app.services.embeddings import EmbeddingResult


class _FakeSessionCtx:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_rag_search_returns_no_results_message(monkeypatch) -> None:
    tool = RagSearchTool()

    async def _fake_embed_text(text: str, model_name=None):
        return EmbeddingResult(vector=[0.1] * 768, model_name="nomic-embed-text")

    async def _fake_search(*args, **kwargs):
        return []

    monkeypatch.setattr(tool.embeddings, "embed_text", _fake_embed_text)
    monkeypatch.setattr("app.mcp_server.tools.rag_search.search_document_chunks", _fake_search)
    monkeypatch.setattr("app.mcp_server.tools.rag_search.AsyncSessionLocal", lambda: _FakeSessionCtx())

    out = asyncio.run(tool.run({"query": "company policy"}))
    assert out == "No relevant information found in knowledge base."


def test_rag_search_formats_citations(monkeypatch) -> None:
    tool = RagSearchTool()

    async def _fake_embed_text(text: str, model_name=None):
        return EmbeddingResult(vector=[0.1] * 768, model_name="nomic-embed-text")

    chunk = SimpleNamespace(
        page_number=3,
        section_heading="Access",
        text="Employees must use VPN.",
    )
    doc = SimpleNamespace(
        filename="manual.pdf",
        created_at=datetime(2026, 4, 13, 10, 0, 0),
    )

    async def _fake_search(*args, **kwargs):
        return [(chunk, doc, 0.91)]

    monkeypatch.setattr(tool.embeddings, "embed_text", _fake_embed_text)
    monkeypatch.setattr("app.mcp_server.tools.rag_search.search_document_chunks", _fake_search)
    monkeypatch.setattr("app.mcp_server.tools.rag_search.AsyncSessionLocal", lambda: _FakeSessionCtx())

    out = asyncio.run(tool.run({"query": "VPN policy", "top_k": 5}))
    assert "Source: manual.pdf (page 3, section Access)" in out
    assert "Similarity: 0.910" in out
    assert "Content: \"Employees must use VPN.\"" in out


def test_rag_search_invalid_query(monkeypatch) -> None:
    tool = RagSearchTool()
    out = asyncio.run(tool.run({"query": ""}))
    assert out == "RAG search not used: missing or invalid query."
