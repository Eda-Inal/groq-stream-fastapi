from typing import AsyncIterator

import httpx

from app.core.config import settings


class GroqClient:
    """
    Async client for interacting with Groq's OpenAI-compatible API.
    """

    def __init__(self) -> None:
        self._base_url: str = settings.groq_base_url
        self._api_key: str = settings.groq_api_key

    async def stream_chat_completion(self) -> AsyncIterator[str]:
        """
        Placeholder for streaming chat completions from Groq.

        Yields:
            str: Streamed text chunks from the model.
        """
        # This will be implemented in the next step
        raise NotImplementedError
