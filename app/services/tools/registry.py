from app.services.tools.calculator import CalculatorTool
from app.services.tools.web_search import WebSearchTool
from app.services.groq_client import GroqClient
from app.core.config import settings
import structlog

logger = structlog.get_logger()

class ToolRegistry:
    def __init__(self) -> None:
        self._tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
        }
        self._router_client = GroqClient()

    async def maybe_apply(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        last_user_message = messages[-1]["content"]
        
        tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in self._tools.values()])
        
        router_prompt = [
            {"role": "system", "content": (
                "You are a strict intent router. Decide if the user's message REQUIRES an external tool.\n\n"
                "TOOLS:\n"
                f"{tool_descriptions}\n\n"
                "RULES:\n"
                "1. If the request is a general question, explanation, or conversation, respond 'NONE'.\n"
                "2. If the LLM can easily answer without a tool (like 5+10), respond 'NONE'.\n"
                "3. Respond ONLY with the tool name or 'NONE'. No punctuation or explanation."
            )},
            {"role": "user", "content": last_user_message}
        ]

        try:
            selected_tool_name = await self._router_client.complete_chat(
                messages=router_prompt,
                model="llama-3.1-8b-instant",
                temperature=0.0
            )
            
            selected_tool_name = selected_tool_name.strip().lower().replace("'", "").replace('"', "")
            logger.info("router_decision", decision=selected_tool_name)

            if selected_tool_name in self._tools:
                tool = self._tools[selected_tool_name]
                result = await tool.run(messages)
                
                new_messages = messages.copy()
                new_messages.insert(-1, {
                    "role": "system",
                    "content": f"[TOOL_RESULT] {result}. Use this information to provide a helpful response."
                })
                return new_messages

        except Exception as e:
            logger.error("routing_failed", error=str(e))
        
        return messages