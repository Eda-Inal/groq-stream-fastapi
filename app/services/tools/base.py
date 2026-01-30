from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """
    Abstract base class for all tools.
    """

    name: str
    description: str

    @abstractmethod
    async def should_run(self, messages: list[dict[str, str]]) -> bool:
        last = messages[-1]
        if last.get("role") != "user":
            return False

        content = last.get("content", "").lower()

        # 1. Must contain at least one number
        has_number = any(char.isdigit() for char in content)

        # 2. Must look like a calculation request
        math_signals = [
            "what is",
            "calculate",
            "compute",
            "result",
            "value",
        ]

        has_signal = any(signal in content for signal in math_signals)

        return has_number and has_signal

    @abstractmethod
    async def run(self, messages: list[dict[str, str]]) -> str:
        """
        Execute the tool and return a textual result
        to be injected back into the conversation.
        """
        raise NotImplementedError
