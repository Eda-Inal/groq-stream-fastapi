from app.services.tools.calculator import CalculatorTool
from app.services.tools.web_search import WebSearchTool


class ToolRegistry:
    def __init__(self) -> None:
        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
        }

    def openai_tools(self) -> list[dict]:
        return [t.openai_schema() for t in self.tools.values()]

    def get(self, name: str):
        # Fail-soft: unknown tool name must not crash
        return self.tools.get(name)
