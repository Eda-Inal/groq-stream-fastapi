import json
from typing import AsyncIterator

import httpx

from app.core.config import settings


class GroqClient:
 

    def __init__(self) -> None:
        self._base_url: str = settings.groq_base_url
        self._api_key: str = settings.groq_api_key

    async def stream_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> AsyncIterator[str]:
  
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model or settings.groq_default_model,
            "messages": messages,
            "stream": True,
        }

        async with httpx.AsyncClient(base_url=self._base_url, timeout=None) as client:
            async with client.stream(
                method="POST",
                url="/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.strip() == "data: [DONE]":
                        break

                    if line.startswith("data: "):
                        data = json.loads(line.removeprefix("data: "))

                        delta = data["choices"][0]["delta"]
                        content = delta.get("content")

                        if content:
                            yield content
