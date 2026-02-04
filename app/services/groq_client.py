import json
from typing import Any, AsyncIterator

import httpx
import structlog

from app.core.config import settings

logger = structlog.get_logger()


class GroqClient:
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
        Stream chat completion from Groq (OpenAI-compatible).

        IMPORTANT:
        - This generator MUST NOT crash on non-standard SSE lines.
        - Some SSE payloads may NOT contain 'choices' (ping, error, etc.).
        - We emit events in a simplified internal format:
          - {"type": "chunk", "text": str|None}
          - {"type": "tool_call", "tool_calls": list[dict]}
          - {"type": "done", "finish_reason": str}
          - {"type": "error", "status": int|None, "message": str}
        """

        log = logger.bind(model=model)

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
            "presence_penalty": presence_penalty,
            "seed": seed,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }

        timeout = httpx.Timeout(settings.groq_read_timeout)

        try:
            async with httpx.AsyncClient(
                base_url=settings.groq_base_url,
                headers=headers,
                timeout=timeout,
                verify=getattr(settings, "groq_verify_ssl", True),
            ) as client:
                async with client.stream(
                    "POST",
                    "/chat/completions",
                    json=payload,
                ) as r:
                    # HTTP-level error: do NOT throw, yield error and finish cleanly
                    if r.status_code >= 400:
                        body = await r.aread()
                        message = body.decode("utf-8", errors="replace") if body else ""
                        log.error("groq_http_error", status=r.status_code, body=message[:500])
                        yield {
                            "type": "error",
                            "status": r.status_code,
                            "message": message or f"Groq HTTP error {r.status_code}",
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

                        # Some providers may send non-JSON keepalives or pings
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            log.debug("groq_sse_non_json", raw=raw[:200])
                            continue

                        # Provider error shapes (no 'choices')
                        if isinstance(data, dict) and "error" in data:
                            err = data.get("error") or {}
                            msg = err.get("message") or str(err) or "Groq streaming error"
                            log.error("groq_stream_error_payload", message=msg)
                            yield {"type": "error", "status": None, "message": msg}
                            yield {"type": "done", "finish_reason": "error"}
                            return

                        # Normal OpenAI-ish payload should have choices
                        choices = data.get("choices") if isinstance(data, dict) else None
                        if not choices:
                            # e.g. {"type":"ping"} or similar
                            continue

                        # Defensive indexing
                        choice0 = choices[0] if isinstance(choices, list) and choices else None
                        if not isinstance(choice0, dict):
                            continue

                        delta = choice0.get("delta") or {}
                        if not isinstance(delta, dict):
                            continue

                        # Tool calls delta
                        if "tool_calls" in delta and delta["tool_calls"] is not None:
                            tool_calls = delta.get("tool_calls")
                            if isinstance(tool_calls, list) and tool_calls:
                                yield {"type": "tool_call", "tool_calls": tool_calls}
                            continue

                        # Content delta (can be None)
                        if "content" in delta:
                            yield {"type": "chunk", "text": delta.get("content")}
                            continue

                        # Finish reason may arrive with empty delta (some providers)
                        finish_reason = choice0.get("finish_reason")
                        if finish_reason:
                            yield {"type": "done", "finish_reason": finish_reason}
                            return

        except httpx.HTTPError as e:
            log.error("groq_network_error", error=str(e))
            yield {"type": "error", "status": None, "message": f"Groq network error: {str(e)}"}
            yield {"type": "done", "finish_reason": "error"}
        except Exception as e:
            log.error("groq_unexpected_error", error=str(e), exc_info=True)
            yield {"type": "error", "status": None, "message": f"Groq unexpected error: {str(e)}"}
            yield {"type": "done", "finish_reason": "error"}
