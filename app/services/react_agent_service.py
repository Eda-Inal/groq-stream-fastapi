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
        """rag_search is the only tool available to the agent."""
        tools = await self.mcp.list_tools()
        return [
            t for t in tools
            if isinstance(t, dict) and t.get("function", {}).get("name") == "rag_search"
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

        The model may call one of the available tools per iteration, observe
        the result, and continue until it produces a Final Answer or the
        iteration budget is exhausted.
        """
        log = logger.bind(model=model, conversation_id=conversation_id, user_id=user_id)

        available_tools = await self._get_available_tools()
        available_names = {
            t.get("function", {}).get("name") for t in available_tools
        }

        system_prompt = build_system_prompt(available_tools)
        llm_messages = [{"role": "system", "content": system_prompt}, *messages]

        log.info("agent_loop_started", tool_count=len(available_tools))

        final_answer: str | None = None
        last_raw_response = ""

        for iteration in range(settings.agent_max_iterations):
            iter_log = log.bind(iteration=iteration + 1)

            buffer: list[str] = []
            native_tool_call: dict | None = None
            failed_generation: str | None = None
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
                elif event["type"] == "tool_call":
                    calls = event.get("tool_calls", [])
                    if calls and native_tool_call is None:
                        native_tool_call = calls[0]
                elif event["type"] == "error":
                    # Groq returns 400 when the model generates a native tool call
                    # but no tools were declared (tool_choice=none). The actual tool
                    # call is recoverable from failed_generation in the error body.
                    fg = event.get("failed_generation")
                    if event.get("status") == 400 and fg:
                        failed_generation = fg
                        iter_log.info("native_tool_call_via_failed_generation", raw=fg[:120])
                    else:
                        iter_log.error("agent_llm_error", message=event.get("message"))
                        yield {"type": "error", "message": event.get("message", "LLM error")}
                        yield {"type": "done", "finish_reason": "error"}
                        return
                elif event["type"] == "done":
                    break

            raw_response = "".join(buffer)

            # Model produced a native tool call (either as a tool_call event or
            # as a 400 failed_generation). Recover the tool name + args and
            # append them to whatever text the model already wrote, so the
            # ReAct parser finds a complete Thought/Action/Action Input block.
            recovered_tool: str | None = None
            recovered_args_str = "{}"

            if native_tool_call is not None:
                import json as _json
                fn = native_tool_call.get("function", {})
                recovered_tool = fn.get("name", "")
                try:
                    raw_args = _json.loads(fn.get("arguments", "{}"))
                    clean_args = {k: v for k, v in raw_args.items()
                                  if k not in ("top_k", "similarity_threshold")}
                    recovered_args_str = _json.dumps(clean_args)
                except (_json.JSONDecodeError, AttributeError):
                    recovered_args_str = "{}"
            elif failed_generation is not None:
                import json as _json
                try:
                    fg_obj = _json.loads(failed_generation)
                    recovered_tool = fg_obj.get("name")
                    raw_args = fg_obj.get("arguments", {})
                    # Drop model-supplied top_k / similarity_threshold so system
                    # defaults (RAG_DEFAULT_TOP_K from .env) apply instead.
                    clean_args = {k: v for k, v in raw_args.items()
                                  if k not in ("top_k", "similarity_threshold")}
                    recovered_args_str = _json.dumps(clean_args)
                except (_json.JSONDecodeError, AttributeError):
                    pass

            if recovered_tool:
                if not raw_response.strip():
                    raw_response = (
                        f"Thought: I need to search the documents for this information.\n"
                        f"Action: {recovered_tool}\n"
                        f"Action Input: {recovered_args_str}"
                    )
                else:
                    # Thought text exists but Action was emitted as native tool call —
                    # append the Action so the parser can find a complete block.
                    raw_response = (
                        f"{raw_response.rstrip()}\n"
                        f"Action: {recovered_tool}\n"
                        f"Action Input: {recovered_args_str}"
                    )
                iter_log.info("native_tool_call_converted_to_react", tool=recovered_tool)

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

            # If model skipped the "Action:" line but wrote "Action Input:" with a
            # query key, infer rag_search (it is the only available tool).
            if action is None and isinstance(action_input, dict) and "query" in action_input:
                action = "rag_search"

            if action == "rag_search" and action in available_names and isinstance(action_input, dict):
                # Strip model-supplied retrieval params so system defaults apply.
                action_input.pop("top_k", None)
                action_input.pop("similarity_threshold", None)

                # Server-side injection -- the model never controls this.
                metadata_filter = action_input.get("metadata_filter")
                if not isinstance(metadata_filter, dict):
                    metadata_filter = {}
                if user_id:
                    metadata_filter["user_id"] = user_id
                action_input["metadata_filter"] = metadata_filter

                yield {"type": "action", "tool": action, "args": action_input}

                tool_result = await self.mcp.call_tool(
                    action, action_input, timeout=settings.agent_rag_timeout
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

        yield {"type": "chunk", "text": final_answer}
        yield {"type": "done", "finish_reason": "stop"}
