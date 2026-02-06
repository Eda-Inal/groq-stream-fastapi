from abc import ABC, abstractmethod
from typing import List, Dict


class MCPClient(ABC):
    """
    Abstract MCP client interface.

    Provides a provider-agnostic layer for tool discovery and execution.
    Implementations MUST follow fail-soft behavior and never raise.
    """

    @abstractmethod
    async def list_tools(self) -> List[Dict]:
        """
        Returns OpenAI-compatible tools schema list.

        Fail-soft:
        - Must return [] on error.
        - Must NOT raise exceptions.
        """
        raise NotImplementedError

    @abstractmethod
    async def call_tool(self, name: str, args: dict) -> str:
        """
        Executes a tool by name.

        Fail-soft:
        - Must NEVER raise exceptions.
        - Must return a string on any error.
        """
        raise NotImplementedError
