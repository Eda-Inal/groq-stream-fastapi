from collections.abc import AsyncIterator
import json

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.chat_log import create_chat_log
from app.services.groq_client import GroqClient
from app.services.tools.registry import ToolRegistry

logger = structlog.get_logger()


FINALIZATION_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You have access to the results of tools you previously chose to call. "
        "Review those results carefully and incorporate your own reasoning, judgment, "
        "and interpretation when forming your final response."
    ),
}


class ChatService:
    def __init__(self) -> None:
        self.client = GroqClient()
        self.tools = ToolRegistry()

    async def stream_chat(
        self,
        *,
        session: AsyncSession,
        messages: list[dict],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[dict]:

        log = logger.bind(model=model)

        effective_messages = list(messages)
        tools_schema = self.tools.openai_tools()

        full_response: list[str] = []

        max_rounds = 8
        max_total_tool_calls = 10
        total_tool_calls_executed = 0
        any_tool_executed = False

        def merge_tool_call_delta(state: dict, deltas: list[dict]) -> None:
            for d in deltas:
                if not isinstance(d, dict):
                    continue
                key = d.get("id") or f"idx:{d.get('index')}"
                if key not in state:
                    state[key] = {
                        "id": d.get("id"),
                        "function": {"name": None, "arguments": ""},
                    }
                fn = d.get("function") or {}
                cur = state[key]["function"]
                if fn.get("name"):
                    cur["name"] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    cur["arguments"] += fn["arguments"]

        def build_openai_tool_calls(state: dict[str, dict]) -> list[dict]:
            tool_calls: list[dict] = []
            fallback_i = 0
            for _, call in state.items():
                fn = call.get("function") or {}
                name = fn.get("name")
                args = fn.get("arguments", "")
                if not name:
                    continue

                call_id = call.get("id")
                if not call_id:
                    fallback_i += 1
                    call_id = f"toolcall_{fallback_i}"

                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args if isinstance(args, str) else "",
                        },
                    }
                )
            return tool_calls

        round_no = 0

        while True:
            round_no += 1
            if round_no > max_rounds:
                break

            tool_state: dict[str, dict] = {}
            buffered_chunks: list[dict] = []
            saw_tool_call = False

            async for event in self.client.stream_chat_completion(
                messages=effective_messages,
                model=model,
                tools=tools_schema,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                seed=seed,
            ):
                etype = event.get("type")

                if etype == "chunk":
                    if event.get("text"):
                        buffered_chunks.append(event)
                    continue

                if etype == "tool_call":
                    saw_tool_call = True
                    merge_tool_call_delta(tool_state, event.get("tool_calls", []))
                    continue

                if etype in ("error", "done"):
                    break

            if not saw_tool_call:
                for ev in buffered_chunks:
                    full_response.append(ev.get("text", ""))
                    yield ev
                break

            openai_tool_calls = build_openai_tool_calls(tool_state)
            if openai_tool_calls:
                effective_messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": openai_tool_calls,
                    }
                )

            for call in openai_tool_calls:
                if total_tool_calls_executed >= max_total_tool_calls:
                    break

                fn = call.get("function") or {}
                name = fn.get("name")
                raw_args = fn.get("arguments", "")
                tool_call_id = call.get("id") or name

                if not name:
                    continue

                try:
                    args = json.loads(raw_args) if raw_args else {}
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    args = {}

                tool = self.tools.get(name)
                if tool is None:
                    result = f"Tool '{name}' is not available."
                else:
                    try:
                        result = await tool.run(args)
                        if not isinstance(result, str):
                            result = str(result)
                    except Exception:
                        result = f"Tool '{name}' failed."

                effective_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result,
                    }
                )

                any_tool_executed = True
                total_tool_calls_executed += 1

            if any_tool_executed:
                effective_messages.append(FINALIZATION_SYSTEM_MESSAGE)

                saw_final = False
                async for event in self.client.stream_chat_completion(
                    messages=effective_messages,
                    model=model,
                    tools=None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    seed=seed,
                ):
                    if event.get("type") == "chunk" and event.get("text"):
                        saw_final = True
                        full_response.append(event["text"])
                    if event.get("type") == "tool_call":
                        log.warning("tool_call_in_finalization_ignored")
                        continue
                    yield event

                if not saw_final:
                    yield {
                        "type": "chunk",
                        "text": "Tool sonuçlarını aldım ancak model final cevabı üretmedi.",
                    }
                    yield {"type": "done", "finish_reason": "stop"}
                break

        await create_chat_log(
            session=session,
            prompt=messages[0]["content"] if messages else "",
            messages=effective_messages,
            response="".join(full_response),
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
        )
        await session.commit()
