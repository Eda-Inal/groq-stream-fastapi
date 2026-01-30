from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """
    Abstract base class for all tools.
    The description is crucial as it's used by the Router LLM 
    to decide when to invoke this tool.
    """

    name: str
    description: str

    @abstractmethod
    async def run(self, messages: list[dict[str, str]]) -> str:
        """
        Execute the tool and return a textual result
        to be injected back into the conversation.
        """
        raise NotImplementedError