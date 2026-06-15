from collections.abc import AsyncIterator

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.repositories.chat_log import create_chat_log
from app.services.groq_client import LLMClient
from app.services.react_agent_prompts import build_system_prompt, parse_react_response
from app.services.tool_client.remote_client import RemoteToolClient

logger = structlog.get_logger()


class ReActAgentService:
    """
    Orchestrates a text-based ReAct (Thought/Action/Observation/Final Answer)
    loop. Independent from ChatService: no native function-calling, no shared
    state with the /chat/stream pipeline.
    """

    def __init__(self) -> None:
        self.client = LLMClient()
        self.mcp = RemoteToolClient()

    async def _get_available_tools(self) -> list[dict]:
        """
        Return the tool schemas the agent may use.

        Stage 2: only web_search is exposed. Returns an empty list (tool-less
        prompt) if web_search is not registered on the tool server.
        """
        tools = await self.mcp.list_tools()
        return [
            t for t in tools
            if isinstance(t, dict) and t.get("function", {}).get("name") == "web_search"
        ]

    async def stream_agent(
        self,
        *,
        session: AsyncSession,
        messages: list[dict],
        model: str,
        user_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        conversation_id: str | None = None,
        tags: list[str] | None = None,
    ) -> AsyncIterator[dict]:
        """
        Run the ReAct loop and yield streaming events.

        Single-tool variant: the model may call web_search at most once per
        iteration, observe the result, and continue until it produces a
        Final Answer or the iteration budget is exhausted.
        """
        log = logger.bind(model=model, conversation_id=conversation_id, user_id=user_id)

        available_tools = await self._get_available_tools()
        web_search_name = (
            available_tools[0].get("function", {}).get("name")
            if available_tools else None
        )

        system_prompt = build_system_prompt(available_tools)
        llm_messages = [{"role": "system", "content": system_prompt}, *messages]

        log.info("agent_loop_started", tool_count=len(available_tools))

        final_answer: str | None = None
        last_raw_response = ""

        for iteration in range(settings.agent_max_iterations):
            iter_log = log.bind(iteration=iteration + 1)

            buffer: list[str] = []
            async for event in self.client.stream_chat_completion(
                messages=llm_messages,
                model=model,
                tools=None,
                temperature=temperature if temperature is not None else 0.0,
                max_tokens=max_tokens,
                stop=["Observation:"],
                call_type="agent",
            ):
                if event["type"] == "chunk":
                    if event.get("text"):
                        buffer.append(event["text"])
                elif event["type"] == "error":
                    iter_log.error("agent_llm_error", message=event.get("message"))
                    yield {"type": "error", "message": event.get("message", "LLM error")}
                    yield {"type": "done", "finish_reason": "error"}
                    return
                elif event["type"] == "done":
                    break

            raw_response = "".join(buffer)
            last_raw_response = raw_response
            parsed = parse_react_response(raw_response)

            iter_log.info(
                "agent_loop_iteration_parsed",
                has_thought=parsed["thought"] is not None,
                action=parsed["action"],
                has_final_answer=parsed["final_answer"] is not None,
            )

            if parsed["thought"]:
                yield {"type": "thought", "text": parsed["thought"]}

            if parsed["final_answer"]:
                final_answer = parsed["final_answer"]
                break

            action = parsed["action"]
            action_input = parsed["action_input"]

            if action == web_search_name and isinstance(action_input, dict):
                yield {"type": "action", "tool": action, "args": action_input}

                tool_result = await self.mcp.call_tool(
                    action, action_input, timeout=settings.agent_web_search_timeout
                )

                yield {
                    "type": "observation",
                    "tool": action,
                    "result": tool_result.content,
                }

                llm_messages.append({"role": "assistant", "content": raw_response})
                llm_messages.append({"role": "user", "content": f"Observation: {tool_result.content}"})
                continue

            # Unrecognized action or malformed Action Input -- stop the loop
            # with whatever text the model produced. Stage 4 refines this.
            final_answer = raw_response
            break

        if final_answer is None:
            final_answer = last_raw_response

        yield {"type": "chunk", "text": final_answer}
        yield {"type": "done", "finish_reason": "stop"}

        prompt_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                prompt_text = m.get("content", "")
                break

        await create_chat_log(
            session=session,
            prompt=prompt_text,
            messages=messages,
            response=final_answer,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            conversation_id=conversation_id,
            turn_index=None,
            user_id=user_id,
        )
        await session.commit()
