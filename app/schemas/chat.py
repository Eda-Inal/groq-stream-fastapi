"""
Request schemas for chat-related endpoints.

These schemas define the OpenAI-compatible chat message format
used by the streaming chat endpoint.
"""

from typing import Literal
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """
    Represents a single chat message.

    Attributes:
        role: The role of the message sender
              (system, user, or assistant).
        content: The textual content of the message.
    """

    role: Literal["system", "user", "assistant"]
    content: str


class ChatStreamRequest(BaseModel):
    """
    Request body for streaming chat completions.

    Attributes:
        messages: List of chat messages sent to the model.
        model: Optional model override.
    """

    messages: list[ChatMessage]
    model: str | None = None
