from pydantic import BaseModel, Field
from typing import List, Optional


class BulkChatItem(BaseModel):
    messages: List[dict]
    model: Optional[str] = None
    temperature: Optional[float] = 0.7



class BulkChatRequest(BaseModel):
    items: List[BulkChatItem] = Field(
        ..., description="List of independent chat requests"
    )
