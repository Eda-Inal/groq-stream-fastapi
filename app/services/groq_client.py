import json
import re
from typing import Any, AsyncIterator

import httpx
import structlog

from app.core.config import settings, AVAILABLE_MODELS

logger = structlog.get_logger()

_PROVIDER_CONFIGS = {
    "groq": lambda: (settings.groq_base_url, settings.groq_api_key),
    "openrouter": lambda: (settings.openrouter_base_url, settings.openrouter_api_key),
    "gemini": lambda: (settings.gemini_base_url, settings.gemini_api_key),
    "sambanova": lambda: (settings.sambanova_base_url, settings.sambanova_api_key),
}


def _resolve_provider(model: str) -> tuple[str, str]:
    """Return (base_url, api_key) for the given model."""
    model_info = AVAILABLE_MODELS.get(model, {})
    provider = model_info.get("provider", "groq")

    factory = _PROVIDER_CONFIGS.get(provider)
    if factory is None:
        raise ValueError(f"Unknown provider '{provider}' for model '{model}'")

    base_url, api_key = factory()

    if not api_key:
        raise ValueError(
            f"API key for provider '{provider}' is not configured. "
            f"Set the corresponding environment variable "
            f"(GROQ_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY)."
        )

    return base_url, api_key


class LLMClient:
    def __init__(self) -> None:
        # Cache key includes api_key to avoid returning a client created with
        # stale credentials when key rotates.
        self._clients: dict[tuple[str, str], httpx.AsyncClient] = {}

    def _get_client(self, base_url: str, api_key: str) -> httpx.AsyncClient:
        """Return a shared httpx client per (base_url, api_key) to enable connection reuse."""
        cache_key = (base_url, api_key)
        existing = self._clients.get(cache_key)
        if existing is not None and not existing.is_closed:
            return existing
        client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(settings.groq_read_timeout),
            verify=getattr(settings, "groq_verify_ssl", True),
        )
        self._clients[cache_key] = client
        return client

    async def stream_chat_completion(
        self,
        *,
        messages: list[dict],
        model: str,
        tools: list[dict] | None = None,
        stream: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
        call_type: str | None = None,
    ) -> AsyncIterator[dict]:
        """
        Stream chat completion from any OpenAI-compatible provider.

        Supported providers: groq, openrouter, gemini.
        Provider is resolved automatically from AVAILABLE_MODELS based on the
        model name. Falls back to groq if the model is not in AVAILABLE_MODELS.

        Emits events in a simplified internal format:
          - {"type": "chunk", "text": str|None}
          - {"type": "tool_call", "tool_calls": list[dict]}
          - {"type": "done", "finish_reason": str}
          - {"type": "error", "status": int|None, "message": str}
        """
        log = logger.bind(model=model)

        try:
            base_url, api_key = _resolve_provider(model)
        except ValueError as exc:
            log.error("llm_provider_config_error", error=str(exc))
            yield {"type": "error", "status": None, "message": str(exc)}
            yield {"type": "done", "finish_reason": "error"}
            return

        model_info = AVAILABLE_MODELS.get(model, {})
        provider = model_info.get("provider", "groq")
        use_stream = stream if stream is not None else model_info.get("stream", True)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": use_stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "seed": seed,
        }
        if use_stream:
            payload["stream_options"] = {"include_usage": True}

        payload = {k: v for k, v in payload.items() if v is not None}

        # Gemini does not support these OpenAI-specific parameters.
        if provider == "gemini":
            for key in ("frequency_penalty", "presence_penalty", "seed", "stream_options"):
                payload.pop(key, None)

        log.info(
            "llm_request",
            provider=provider,
            base_url=base_url,
            stream=use_stream,
            temperature=temperature,
            call_type=call_type,
        )

        # State for filtering <think>...</think> blocks (reasoning models e.g. Qwen3)
        _in_think = False
        _carry = ""  # partial tag carried over from previous chunk

        def _filter_think(text: str) -> str:
            nonlocal _in_think, _carry
            text = _carry + text
            _carry = ""
            out: list[str] = []
            while text:
                if not _in_think:
                    idx = text.find("<think>")
                    if idx == -1:
                        for i in range(min(7, len(text)), 0, -1):
                            if "<think>".startswith(text[-i:]):
                                _carry = text[-i:]
                                text = text[:-i]
                                break
                        out.append(text)
                        break
                    out.append(text[:idx])
                    _in_think = True
                    text = text[idx + 7:]
                else:
                    idx = text.find("</think>")
                    if idx == -1:
                        for i in range(min(8, len(text)), 0, -1):
                            if "</think>".startswith(text[-i:]):
                                _carry = text[-i:]
                                break
                        break
                    _in_think = False
                    text = text[idx + 8:]
            return "".join(out)

        try:
            client = self._get_client(base_url, api_key)

            if not use_stream:
                response = await client.post("/chat/completions", json=payload)
                if response.status_code >= 400:
                    body = response.content.decode("utf-8", errors="replace")
                    log.error("llm_http_error", status=response.status_code, body=body[:500])
                    err: dict = {
                        "type": "error",
                        "status": response.status_code,
                        "message": body or f"HTTP error {response.status_code} from {provider}",
                    }
                    try:
                        body_err = json.loads(body).get("error") or {}
                        if body_err.get("message"):
                            err["message"] = body_err["message"]
                        if body_err.get("failed_generation"):
                            err["failed_generation"] = body_err["failed_generation"]
                    except (json.JSONDecodeError, AttributeError):
                        pass
                    if response.status_code == 429:
                        raw_ra = response.headers.get("retry-after") or response.headers.get("Retry-After")
                        if raw_ra and raw_ra.isdigit():
                            err["retry_after"] = int(raw_ra)
                        else:
                            m = re.search(r'retry in ([\d.]+)s', body)
                            if m:
                                err["retry_after"] = int(float(m.group(1))) + 1
                    yield err
                    yield {"type": "done", "finish_reason": "error"}
                    return

                data = response.json()
                choices = data.get("choices") or []
                choice0 = choices[0] if choices else {}
                message = choice0.get("message") or {}

                tool_calls = message.get("tool_calls")
                if tool_calls:
                    normalized = [
                        {
                            "index": i,
                            "function": {
                                "name": tc.get("function", {}).get("name"),
                                "arguments": tc.get("function", {}).get("arguments", ""),
                            },
                        }
                        for i, tc in enumerate(tool_calls)
                    ]
                    yield {"type": "tool_call", "tool_calls": normalized}

                content = message.get("content")
                if content:
                    filtered = _filter_think(content)
                    if filtered:
                        yield {"type": "chunk", "text": filtered}

                if isinstance(data.get("usage"), dict):
                    u = data["usage"]
                    yield {
                        "type": "usage",
                        "prompt_tokens": u.get("prompt_tokens", 0),
                        "completion_tokens": u.get("completion_tokens", 0),
                        "total_tokens": u.get("total_tokens", 0),
                    }

                yield {"type": "done", "finish_reason": choice0.get("finish_reason", "stop")}
                return

            async with client.stream(
                "POST",
                "/chat/completions",
                json=payload,
            ) as r:
                if r.status_code >= 400:
                    body = await r.aread()
                    message = body.decode("utf-8", errors="replace") if body else ""
                    log.error("llm_http_error", status=r.status_code, body=message[:500])
                    err: dict = {
                        "type": "error",
                        "status": r.status_code,
                        "message": message or f"HTTP error {r.status_code} from {provider}",
                    }
                    if r.status_code == 429:
                        raw_ra = r.headers.get("retry-after") or r.headers.get("Retry-After")
                        if raw_ra and raw_ra.isdigit():
                            err["retry_after"] = int(raw_ra)
                        else:
                            m = re.search(r'retry in ([\d.]+)s', message)
                            if m:
                                err["retry_after"] = int(float(m.group(1))) + 1
                    yield err
                    yield {"type": "done", "finish_reason": "error"}
                    return

                async for line in r.aiter_lines():
                    if not line:
                        continue

                    line = line.strip()

                    if line == "data: [DONE]":
                        yield {"type": "done", "finish_reason": "stop"}
                        return

                    if not line.startswith("data: "):
                        continue

                    raw = line[6:].strip()
                    if not raw:
                        continue

                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        log.debug("llm_sse_non_json", raw=raw[:200])
                        continue

                    if isinstance(data, dict) and "error" in data:
                        err = data.get("error") or {}
                        msg = err.get("message") or str(err) or "Streaming error"
                        log.error(
                            "llm_stream_error_payload",
                            message=msg,
                            error_type=err.get("type"),
                            error_code=err.get("code"),
                            failed_generation=err.get("failed_generation"),
                        )
                        error_event: dict = {"type": "error", "status": None, "message": msg}
                        if err.get("failed_generation"):
                            error_event["failed_generation"] = err["failed_generation"]
                        yield error_event
                        yield {"type": "done", "finish_reason": "error"}
                        return

                    choices = data.get("choices") if isinstance(data, dict) else None

                    # Final usage-only chunk (stream_options.include_usage=true):
                    # choices is an empty list and usage is populated here.
                    if not choices:
                        if isinstance(data.get("usage"), dict):
                            u = data["usage"]
                            yield {
                                "type": "usage",
                                "prompt_tokens": u.get("prompt_tokens", 0),
                                "completion_tokens": u.get("completion_tokens", 0),
                                "total_tokens": u.get("total_tokens", 0),
                            }
                        continue

                    choice0 = choices[0] if isinstance(choices, list) and choices else None
                    if not isinstance(choice0, dict):
                        continue

                    delta = choice0.get("delta") or {}
                    if not isinstance(delta, dict):
                        continue

                    if "tool_calls" in delta and delta["tool_calls"] is not None:
                        tool_calls = delta.get("tool_calls")
                        if isinstance(tool_calls, list) and tool_calls:
                            yield {"type": "tool_call", "tool_calls": tool_calls}
                        continue

                    if "content" in delta:
                        raw = delta.get("content") or ""
                        filtered = _filter_think(raw)
                        if filtered:
                            yield {"type": "chunk", "text": filtered}
                        continue

                    finish_reason = choice0.get("finish_reason")
                    if finish_reason:
                        if isinstance(data.get("usage"), dict):
                            u = data["usage"]
                            yield {
                                "type": "usage",
                                "prompt_tokens": u.get("prompt_tokens", 0),
                                "completion_tokens": u.get("completion_tokens", 0),
                                "total_tokens": u.get("total_tokens", 0),
                            }
                        yield {"type": "done", "finish_reason": finish_reason}
                        return

        except httpx.HTTPError as e:
            log.error("llm_network_error", error=str(e))
            yield {"type": "error", "status": None, "message": f"Network error ({provider}): {str(e)}"}
            yield {"type": "done", "finish_reason": "error"}
        except Exception as e:
            log.error("llm_unexpected_error", error=str(e), exc_info=True)
            yield {"type": "error", "status": None, "message": f"Unexpected error ({provider}): {str(e)}"}
            yield {"type": "done", "finish_reason": "error"}


# Backward-compatible alias
GroqClient = LLMClient
