"""
Request schemas for the ReAct agent endpoint.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.chat import ChatMessage


class AgentStreamRequest(BaseModel):
    messages: List[ChatMessage]
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    tags: Optional[List[str]] = None
