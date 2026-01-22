"""
Request schemas for chat-related endpoints.

These schemas define the OpenAI-compatible chat message format
used by the streaming chat endpoint.
"""

from typing import Literal,List, Optional, Union
from pydantic import BaseModel, Field 


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
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None