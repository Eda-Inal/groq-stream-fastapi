from __future__ import annotations

import tiktoken

from app.core.chunking_config import TIKTOKEN_ENCODING

_encoding: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return _encoding


def count_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        return len(_get_encoding().encode(text))
    except Exception:
        return 0


def estimate_messages_tokens(messages: list[dict]) -> int:
    """
    Approximate chat token usage including per-message overhead.
    """
    total = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        # Rough OpenAI-style overhead
        total += 4
        role = m.get("role")
        if isinstance(role, str):
            total += count_tokens(role)

        content = m.get("content")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        total += count_tokens(text)

        name = m.get("name")
        if isinstance(name, str):
            total += count_tokens(name)

        tool_calls = m.get("tool_calls")
        if isinstance(tool_calls, list):
            total += count_tokens(str(tool_calls))
    return total + 2


def truncate_text_to_token_budget(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    try:
        ids = _get_encoding().encode(text)
        return _get_encoding().decode(ids[:max_tokens]).strip()
    except Exception:
        return text[: max_tokens * 4].strip()


def truncate_rag_chunks(chunks: list[str], budget: int) -> list[str]:
    """
    Keep chunks in order until budget is exhausted.
    """
    if budget <= 0:
        return []
    out: list[str] = []
    used = 0
    for chunk in chunks:
        tokens = count_tokens(chunk)
        if used + tokens <= budget:
            out.append(chunk)
            used += tokens
            continue
        remaining = budget - used
        if remaining <= 0:
            break
        trimmed = truncate_text_to_token_budget(chunk, remaining)
        if trimmed:
            out.append(trimmed)
        break
    return out
