from typing import Literal, Optional
from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)
    model: Optional[str] = None
    user_id: Optional[str] = None
    language: Literal["en", "tr"] = "en"
