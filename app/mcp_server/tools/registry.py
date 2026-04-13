from app.mcp_server.tools.calculator import CalculatorTool
from app.mcp_server.tools.rag_search import RagSearchTool
from app.mcp_server.tools.web_search import WebSearchTool


class ToolRegistry:
    def __init__(self) -> None:
        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
            "rag_search": RagSearchTool(),
        }

    def openai_tools(self) -> list[dict]:
        return [t.openai_schema() for t in self.tools.values()]

    def get(self, name: str):
        return self.tools.get(name)
