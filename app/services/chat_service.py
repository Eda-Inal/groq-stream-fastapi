from collections.abc import AsyncIterator
import json

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.chat_log import create_chat_log, list_chat_logs_by_conversation
from app.services.groq_client import GroqClient
from app.services.mcp.remote_client import RemoteMCPClient
from app.core.config import settings
from app.utils.token_counter import (
    estimate_messages_tokens,
    truncate_rag_chunks,
    truncate_text_to_token_budget,
    count_tokens,
)

logger = structlog.get_logger()


RAG_TOOL_CALL_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You have access to a private knowledge base via the rag_search tool. "
        "Before answering questions about uploaded documents, internal policies, company/project "
        "facts, or specific factual claims that may depend on private context, ALWAYS call rag_search first. "
        "You may skip rag_search only for general world knowledge, pure math, coding help, or casual chat. "
        "If rag_search returns no relevant results, explicitly say no relevant information was found and do not hallucinate."
    ),
}


FINALIZATION_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "Tool results are available above. Carefully read every tool message whose Content field "
        "contains retrieved passages. Extract the answer directly from those passages and cite the "
        "source filename. If EVERY tool result explicitly says 'No relevant information found', "
        "then—and only then—state that no information was found. Do not say 'not found' if any "
        "tool result contains relevant text."
    ),
}


class ChatService:
    def __init__(self) -> None:
        self.client = GroqClient()
        self.mcp = RemoteMCPClient()

        if not settings.mcp_server_url:
            logger.error("mcp_server_url_missing")

        logger.info(
            "mcp_client_selected",
            mode="remote",
            url=settings.mcp_server_url,
        )

    def _apply_context_budget(
        self,
        messages: list[dict],
        *,
        max_input_tokens: int,
        rag_tool_budget: int,
    ) -> list[dict]:
        """
        Trim message history/tool payloads to fit model input budget.
        """
        trimmed = [dict(m) if isinstance(m, dict) else m for m in messages]

        # 1) Trim rag_search tool payload first, keeping top chunks.
        for m in trimmed:
            if not isinstance(m, dict):
                continue
            if m.get("role") != "tool" or m.get("name") != "rag_search":
                continue
            content = m.get("content")
            if not isinstance(content, str):
                continue
            blocks = [b.strip() for b in content.split("\n---\n") if b.strip()]
            if not blocks:
                continue
            kept = truncate_rag_chunks(blocks, rag_tool_budget)
            m["content"] = "\n---\n".join(kept)

        # 2) Remove oldest non-system messages until under budget.
        def _over_budget() -> bool:
            return estimate_messages_tokens(trimmed) > max_input_tokens

        idx = 0
        while _over_budget() and idx < len(trimmed):
            m = trimmed[idx]
            if isinstance(m, dict) and m.get("role") not in ("system",):
                trimmed.pop(idx)
                continue
            idx += 1

        # 3) Last resort: trim longest tool content.
        if _over_budget():
            tool_indices = []
            for i, m in enumerate(trimmed):
                if isinstance(m, dict) and m.get("role") == "tool" and isinstance(m.get("content"), str):
                    tool_indices.append((i, count_tokens(m["content"])))
            tool_indices.sort(key=lambda x: x[1], reverse=True)
            for i, _ in tool_indices:
                excess = estimate_messages_tokens(trimmed) - max_input_tokens
                if excess <= 0:
                    break
                current = trimmed[i]["content"]
                keep = max(0, count_tokens(current) - excess - 16)
                trimmed[i]["content"] = truncate_text_to_token_budget(current, keep)

        return trimmed

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
        conversation_id: str | None = None,
    ) -> AsyncIterator[dict]:

        log = logger.bind(model=model)

        effective_messages = list(messages)

        history_logs = []
        turn_index = None

        if conversation_id:
            history_logs = await list_chat_logs_by_conversation(
                session=session,
                conversation_id=conversation_id,
                limit=20,
            )

            history_messages: list[dict] = []
            for log_item in history_logs:
                msgs = log_item.messages
                if not isinstance(msgs, list):
                    continue

                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    role = m.get("role")
                    content = m.get("content")
                    if (
                        role in ("user", "assistant")
                        and isinstance(content, str)
                        and "role" in m
                        and "content" in m
                    ):
                        history_messages.append(m)

            if history_messages:
                effective_messages = history_messages + effective_messages

        tools_schema = await self.mcp.list_tools()
        rag_available = any(
            isinstance(t, dict)
            and isinstance(t.get("function"), dict)
            and t["function"].get("name") == "rag_search"
            for t in tools_schema
        )

        if settings.rag_system_prompt_enabled and rag_available:
            already_guided = any(
                isinstance(m, dict)
                and m.get("role") == "system"
                and isinstance(m.get("content"), str)
                and "private knowledge base via the rag_search tool" in m["content"]
                for m in effective_messages
            )
            if not already_guided:
                effective_messages = [RAG_TOOL_CALL_SYSTEM_MESSAGE] + effective_messages

        full_response: list[str] = []

        max_rounds = 8
        max_total_tool_calls = 10
        total_tool_calls_executed = 0
        any_tool_executed = False

        def merge_tool_call_delta(state: dict, deltas: list[dict]) -> None:
            for d in deltas:
                if not isinstance(d, dict):
                    continue
                # Use index as the stable key; id arrives only in the first delta.
                key = str(d.get("index", 0))
                if key not in state:
                    state[key] = {
                        "id": None,
                        "function": {"name": None, "arguments": ""},
                    }
                # Persist the call id from whichever delta carries it.
                if d.get("id"):
                    state[key]["id"] = d["id"]
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

                result = await self.mcp.call_tool(name, args)

                effective_messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "tool_call_id": tool_call_id,
                        "content": result,
                    }
                )

                any_tool_executed = True
                total_tool_calls_executed += 1

            if any_tool_executed:
                if settings.rag_system_prompt_enabled:
                    effective_messages.append(FINALIZATION_SYSTEM_MESSAGE)

                reserve = max(0, max_tokens or settings.response_reserve_tokens)
                max_input_tokens = max(500, settings.max_context_tokens - reserve)
                effective_messages = self._apply_context_budget(
                    effective_messages,
                    max_input_tokens=max_input_tokens,
                    rag_tool_budget=settings.rag_tool_max_context_tokens,
                )

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

        if conversation_id:
            max_turn = 0
            for log_item in history_logs:
                if log_item.turn_index and log_item.turn_index > max_turn:
                    max_turn = log_item.turn_index
            turn_index = max_turn + 1 if max_turn else 1
        else:
            turn_index = None

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
            conversation_id=conversation_id,
            turn_index=turn_index,
        )
        await session.commit()
