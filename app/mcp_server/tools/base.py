from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """
    Abstract base class for all tools.
    """

    name: str
    description: str
    parameters: dict

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def run(self, args: dict[str, Any]) -> str:
        raise NotImplementedError
