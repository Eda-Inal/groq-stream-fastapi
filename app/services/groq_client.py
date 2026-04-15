import json
from typing import Any, AsyncIterator

import httpx
import structlog

from app.core.config import settings, AVAILABLE_MODELS

logger = structlog.get_logger()

_PROVIDER_CONFIGS = {
    "groq": lambda: (settings.groq_base_url, settings.groq_api_key),
    "openrouter": lambda: (settings.openrouter_base_url, settings.openrouter_api_key),
    "gemini": lambda: (settings.gemini_base_url, settings.gemini_api_key),
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
    async def stream_chat_completion(
        self,
        *,
        messages: list[dict],
        model: str,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
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

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "seed": seed,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        timeout = httpx.Timeout(settings.groq_read_timeout)

        provider = AVAILABLE_MODELS.get(model, {}).get("provider", "groq")
        log.info("llm_request", provider=provider, base_url=base_url)

        try:
            async with httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=timeout,
                verify=getattr(settings, "groq_verify_ssl", True),
            ) as client:
                async with client.stream(
                    "POST",
                    "/chat/completions",
                    json=payload,
                ) as r:
                    if r.status_code >= 400:
                        body = await r.aread()
                        message = body.decode("utf-8", errors="replace") if body else ""
                        log.error("llm_http_error", status=r.status_code, body=message[:500])
                        yield {
                            "type": "error",
                            "status": r.status_code,
                            "message": message or f"HTTP error {r.status_code} from {provider}",
                        }
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
                            log.error("llm_stream_error_payload", message=msg)
                            yield {"type": "error", "status": None, "message": msg}
                            yield {"type": "done", "finish_reason": "error"}
                            return

                        choices = data.get("choices") if isinstance(data, dict) else None
                        if not choices:
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
                            yield {"type": "chunk", "text": delta.get("content")}
                            continue

                        finish_reason = choice0.get("finish_reason")
                        if finish_reason:
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
