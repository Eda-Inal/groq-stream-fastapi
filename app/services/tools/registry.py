from app.services.tools.calculator import CalculatorTool
import structlog

logger = structlog.get_logger()

class ToolRegistry:
    def __init__(self) -> None:
        self._tools = [CalculatorTool()]

    async def maybe_apply(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        for tool in self._tools:
            if await tool.should_run(messages):
                logger.info("tool_invoked", tool=tool.name)
                result = await tool.run(messages)
                
                # Insert the result as a system instruction right before the last user message
                new_messages = messages.copy()
                new_messages.insert(-1, {
                    "role": "system",
                    "content": f"[TOOL_RESULT] {result}. Use this value to answer the user's question accurately."
                })
                return new_messages
        return messages