from collections.abc import AsyncIterator
import asyncio
import json
import re
import time

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.chat_log import create_chat_log, list_chat_logs_by_conversation
from app.mcp_server.tools.base import ToolResult
from app.schemas.chat_bulk import BulkChatItem
from app.services.groq_client import LLMClient
from app.services.mcp.remote_client import RemoteMCPClient
from app.services import tracing as ls
from app.core.config import settings, AVAILABLE_MODELS, FALLBACK_CHAIN
from app.utils.token_counter import (
    estimate_messages_tokens,
    truncate_rag_chunks,
    truncate_text_to_token_budget,
    count_tokens,
)

logger = structlog.get_logger()


ROUTING_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a routing agent. Your ONLY task is to decide whether to call a tool. "
        "Do NOT write any text response — only make tool calls if needed.\n\n"
        "Available tools:\n"
        "1. rag_search — user's private uploaded documents. Use ONLY when the "
        "question explicitly references the user's documents, or when the topic "
        "is clearly internal/private (policies, contracts, personal reports).\n"
        "2. web_search — Use when the answer could have changed recently. This includes: "
        "prices, costs, exchange rates, software versions, population figures, "
        "weather, current role holders (CEO, president, etc.), recent news or events. "
        "Do NOT use for: historical facts, scientific constants, geography, definitions.\n"
        "3. calculator — ANY arithmetic the user explicitly asks to compute, "
        "regardless of how easy the numbers look (e.g. 'what is 18% of 1250'). "
        "Never compute in your head — always route through this tool.\n\n"
        "Rules:\n"
        "- If a tool is needed: call it. Output no text.\n"
        "- You may call multiple tools when a question requires it.\n"
        "- For static general-knowledge questions (historical dates, geography, "
        "definitions, classical authors, established formulas): output nothing. "
        "A separate step will answer.\n"
        "- For simple conversational messages (greetings, thanks): output nothing.\n"
        "- Choose the minimum number of tools required."
    ),
}

DIRECT_ANSWER_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "Answer the user's question clearly and concisely from your own knowledge. "
        "Be accurate, helpful, and direct. "
        "Do not reference any tools or documents."
    ),
}


FINALIZATION_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "Tool results are available above. Read every tool message carefully before answering. "
        "If rag_search returned passages: base your answer exclusively on those passages. "
        "Do not use your training knowledge to override or supplement the retrieved content. "
        "If web_search returned results: summarise the relevant information. "
        "If rag_search returned nothing: respond with "
        "'This information was not found in your documents.' and do not search the web. "
        "Always end your response with a source line: "
        "use 'Source: [filename]' for document answers, 'Source: [URL]' for web answers, "
        "'Source: calculator' for calculator results. "
        "Do not write function calls as text in your response."
    ),
}


class ChatService:
    def __init__(self) -> None:
        self.client = LLMClient()
        self.mcp = RemoteMCPClient()

        if not settings.mcp_server_url:
            logger.error("mcp_server_url_missing")

        logger.info(
            "mcp_client_selected",
            mode="remote",
            url=settings.mcp_server_url,
        )

    @staticmethod
    def _next_fallback(current: str) -> str | None:
        try:
            idx = FALLBACK_CHAIN.index(current)
            return FALLBACK_CHAIN[idx + 1] if idx + 1 < len(FALLBACK_CHAIN) else None
        except ValueError:
            return None

    @staticmethod
    def _is_protected(m: dict, tail_start: int, idx: int) -> bool:
        """Messages in the current turn (from tail_start onward) are protected
        from eviction so the LLM always sees the user question, tool calls /
        results, and finalization prompt."""
        if idx >= tail_start:
            return True
        if m.get("role") == "system":
            return True
        return False

    def _apply_context_budget(
        self,
        messages: list[dict],
        *,
        max_input_tokens: int,
        rag_tool_budget: int,
    ) -> list[dict]:
        """
        Trim message history/tool payloads to fit model input budget.

        Protected (never evicted): system messages and the *current turn*
        which is everything from the last user message onward — i.e. the
        user question, assistant tool-call, tool results, and finalization
        prompt.  Only conversation **history** preceding the current turn
        is eligible for removal.
        """
        trimmed = [dict(m) if isinstance(m, dict) else m for m in messages]

        # Locate the start of the current turn: the last user message.
        tail_start = 0
        for i in range(len(trimmed) - 1, -1, -1):
            m = trimmed[i]
            if isinstance(m, dict) and m.get("role") == "user":
                tail_start = i
                break

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

        # 2) Remove oldest **history** messages (before current turn).
        def _over_budget() -> bool:
            return estimate_messages_tokens(trimmed) > max_input_tokens

        idx = 0
        while _over_budget() and idx < len(trimmed):
            m = trimmed[idx]
            if isinstance(m, dict) and not self._is_protected(m, tail_start, idx):
                trimmed.pop(idx)
                tail_start -= 1
                continue
            idx += 1

        # 3) Last resort: trim longest tool content (even in current turn).
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

    async def _traced_llm_call(
        self,
        *,
        parent_run,
        name: str,
        metadata: dict,
        messages: list[dict],
        model: str,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[dict]:
        """
        Thin async-generator wrapper around LLMClient.stream_chat_completion that
        creates a child LangSmith 'llm' span, collects text + token usage, and
        ends the span when the stream closes (normally or via break/aclose).

        'usage' events from groq_client are consumed internally and never yielded
        to callers — they only feed the LangSmith span.
        """
        llm_run = ls.create_run(
            name=name,
            run_type="llm",
            inputs={
                "messages": messages,
                "model": model,
                "tools": [t.get("function", {}).get("name") for t in (tools or [])],
            },
            parent_run=parent_run,
            metadata=metadata,
            tags=[model],
        )

        text_parts: list[str] = []
        tool_calls_seen: list[dict] = []
        usage: dict = {}
        error_msg: str | None = None

        try:
            async for event in self.client.stream_chat_completion(
                messages=messages,
                model=model,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                seed=seed,
            ):
                etype = event.get("type")
                if etype == "usage":
                    usage = event
                    continue  # consume internally; callers don't need this event
                if etype == "chunk" and event.get("text"):
                    text_parts.append(event["text"])
                elif etype == "tool_call":
                    tool_calls_seen.extend(event.get("tool_calls", []))
                elif etype == "error":
                    error_msg = event.get("message")
                yield event
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            out: dict = {"text": "".join(text_parts)}
            if tool_calls_seen:
                out["tool_calls"] = tool_calls_seen
            if usage:
                out["prompt_tokens"] = usage.get("prompt_tokens")
                out["completion_tokens"] = usage.get("completion_tokens")
                out["total_tokens"] = usage.get("total_tokens")
            ls.end_run(llm_run, outputs=out, error=error_msg)

    async def stream_chat(
        self,
        *,
        session: AsyncSession,
        messages: list[dict],
        model: str,
        user_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
        conversation_id: str | None = None,
        allow_fallback: bool = True,
    ) -> AsyncIterator[dict]:

        log = logger.bind(model=model)
        start_time = time.monotonic()

        # Top-level LangSmith trace for the entire chat turn.
        root_run = ls.create_run(
            name="chat.stream",
            run_type="chain",
            inputs={
                "messages": messages,
                "model": model,
                "user_id": user_id,
                "conversation_id": conversation_id,
            },
            metadata={
                k: v for k, v in {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "seed": seed,
                }.items() if v is not None
            },
            tags=[model],
        )

        effective_messages = list(messages)
        history_logs = []
        turn_index = None
        full_response: list[str] = []
        max_rounds = 8
        max_total_tool_calls = 10
        total_tool_calls_executed = 0
        any_tool_executed = False
        last_rag_result: ToolResult | None = None
        round_no = 0
        active_model = model

        try:
            if conversation_id:
                history_logs = await list_chat_logs_by_conversation(
                    session=session,
                    conversation_id=conversation_id,
                    user_id=user_id,
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
            # Filter tools based on feature flags so individual tools can be
            # disabled via .env without touching the MCP server registry.
            _disabled = set()
            if not settings.web_search_enabled:
                _disabled.add("web_search")
            if not settings.calculator_enabled:
                _disabled.add("calculator")
            if _disabled:
                tools_schema = [
                    t for t in tools_schema
                    if not (
                        isinstance(t, dict)
                        and isinstance(t.get("function"), dict)
                        and t["function"].get("name") in _disabled
                    )
                ]
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
                    and "You are a routing agent" in m["content"]
                    for m in effective_messages
                )
                if not already_guided:
                    effective_messages = [ROUTING_SYSTEM_MESSAGE] + effective_messages

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

            while True:
                round_no += 1
                if round_no > max_rounds:
                    break

                tool_state: dict[str, dict] = {}
                buffered_chunks: list[dict] = []
                saw_tool_call = False
                tool_call_generation_failed = False
                hit_rate_limit = False
                rate_limit_retry_after: int | None = None
                detected_tool: str | None = None

                # ── LLM call 1: tool routing pass ──────────────────────────────
                async for event in self._traced_llm_call(
                    parent_run=root_run,
                    name=f"llm.{active_model}.tool_routing",
                    metadata={
                        "round": round_no,
                        "with_tools": True,
                        "tool_count": len(tools_schema),
                    },
                    messages=effective_messages,
                    model=active_model,
                    tools=tools_schema,
                    temperature=0,
                    max_tokens=256,
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

                    if etype == "error":
                        if event.get("status") == 429:
                            hit_rate_limit = True
                            rate_limit_retry_after = event.get("retry_after")
                            break
                        msg = str(event.get("message", "")).lower()
                        if "call a function" in msg or "failed_generation" in msg:
                            tool_call_generation_failed = True
                            # Try to extract the intended tool name from the error message.
                            # Groq sometimes includes it (e.g. "failed to call web_search").
                            for _tool in ("web_search", "rag_search", "calculator"):
                                if _tool in msg:
                                    detected_tool = _tool
                                    break
                        break

                    if etype == "done":
                        break

                if hit_rate_limit:
                    # RPM (per-minute limit): retry_after is small, wait and retry same model.
                    if rate_limit_retry_after is not None and rate_limit_retry_after <= 60:
                        log.warning("rate_limit_rpm_waiting", model=active_model, retry_after=rate_limit_retry_after)
                        await asyncio.sleep(rate_limit_retry_after + 1)
                        round_no -= 1
                        continue
                    # RPD (daily limit) or unknown: switch to fallback model if available, else return error.
                    if allow_fallback:
                        next_m = self._next_fallback(active_model)
                        if next_m:
                            log.warning("rate_limit_fallback", from_model=active_model, to_model=next_m)
                            active_model = next_m
                            round_no -= 1
                            continue
                    yield {"type": "error", "status": 429, "message": f"Rate limit reached on model '{active_model}'."}
                    yield {"type": "done", "finish_reason": "error"}
                    break

                if not saw_tool_call:
                    # Detect Groq tool-call serialization failure via <|python_tag|> leaking
                    # into chunk text — Groq couldn't build a proper tool_call event so the
                    # model wrote the function call syntax as plain text instead.
                    buffered_text = "".join(ev.get("text", "") for ev in buffered_chunks)
                    if not tool_call_generation_failed and "<|python_tag|>" in buffered_text:
                        tool_call_generation_failed = True

                    # Parse the intended tool name from the leaked tag so the fallback
                    # can call the right tool rather than always defaulting to rag_search.
                    detected_tool: str | None = None
                    if tool_call_generation_failed:
                        tag_match = re.search(r"<\|python_tag\|>(\w+)", buffered_text)
                        if tag_match:
                            detected_tool = tag_match.group(1)

                    if tool_call_generation_failed:
                        log.warning(
                            "tool_call_generation_failed_retrying_without_tools",
                            detected_tool=detected_tool,
                        )

                        last_user_msg = next(
                            (
                                m for m in reversed(effective_messages)
                                if isinstance(m, dict) and m.get("role") == "user"
                                and isinstance(m.get("content"), str)
                            ),
                            None,
                        )

                        if last_user_msg:
                            query = last_user_msg["content"]

                            if detected_tool in ("web_search", None):
                                # detected_tool=None means Groq failed but we don't know which
                                # tool was intended — try web_search first as the most common case.
                                web_result = await self.mcp.call_tool("web_search", {"query": query})
                                if web_result.ok and web_result.content.strip():
                                    effective_messages = list(effective_messages) + [
                                        {
                                            "role": "system",
                                            "content": (
                                                "Web search results:\n"
                                                f"{web_result.content}\n\n"
                                                "Summarise the relevant information. "
                                                "End your response with: Source: [URL]"
                                            ),
                                        }
                                    ]
                                    log.info(
                                        "web_search_injected_after_tool_call_failure",
                                        detected_tool=detected_tool,
                                    )

                            elif rag_available:
                                rag_args: dict = {"query": query}
                                if user_id:
                                    rag_args["metadata_filter"] = {"user_id": user_id}
                                rag_result = await self.mcp.call_tool("rag_search", rag_args)
                                rag_found = (
                                    rag_result.ok
                                    and rag_result.content.strip()
                                    and "No relevant information" not in rag_result.content
                                )
                                if rag_found:
                                    effective_messages = list(effective_messages) + [
                                        {
                                            "role": "system",
                                            "content": (
                                                "Retrieved context from the knowledge base:\n"
                                                f"{rag_result.content}\n\n"
                                                "Answer ONLY using the retrieved context above. "
                                                "Do not use your training knowledge to override these values. "
                                                "Use the exact values from the text. "
                                                "End your response with: Source: [filename]"
                                            ),
                                        }
                                    ]
                                    log.info("rag_context_injected_after_tool_call_failure")
                                else:
                                    effective_messages = list(effective_messages) + [
                                        {
                                            "role": "system",
                                            "content": (
                                                "rag_search returned no relevant results. "
                                                "Tell the user: 'I could not find this in your documents. "
                                                "Would you like me to search the web?'"
                                            ),
                                        }
                                    ]
                                    log.info("rag_empty_asking_user_for_web_search")

                        # Strip the routing prompt so the fallback LLM does not
                        # try to emit tool calls (which would produce <|python_tag|>
                        # again). Replace it with a direct-answer instruction.
                        fallback_messages = [
                            m for m in effective_messages
                            if not (
                                isinstance(m, dict)
                                and m.get("role") == "system"
                                and isinstance(m.get("content"), str)
                                and "You are a routing agent" in m["content"]
                            )
                        ]
                        fallback_messages = [DIRECT_ANSWER_SYSTEM_MESSAGE] + fallback_messages

                        fallback_chunks: list[dict] = []
                        retry_final_event: dict | None = None

                        # ── LLM call 2: fallback retry without tools ────────────
                        async for event in self._traced_llm_call(
                            parent_run=root_run,
                            name=f"llm.{active_model}.fallback_no_tools",
                            metadata={
                                "round": round_no,
                                "with_tools": False,
                                "reason": "tool_call_generation_failed",
                            },
                            messages=fallback_messages,
                            model=active_model,
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
                                fallback_chunks.append(event)
                            elif event.get("type") in ("error", "done"):
                                retry_final_event = event
                                break

                        for ev in fallback_chunks:
                            full_response.append(ev.get("text", ""))
                            yield ev

                        if retry_final_event:
                            yield retry_final_event
                        break

                    # ── Phase 2: direct answer (no tool needed) ─────────────────
                    # LLM 1 (routing) produced no tool call — its buffered text is
                    # discarded. A fresh call with a clean answer prompt ensures the
                    # model focuses solely on answering, not on tool selection.
                    effective_messages.append(DIRECT_ANSWER_SYSTEM_MESSAGE)

                    _da_done = False
                    while not _da_done:
                        _da_429 = False
                        _da_event: dict | None = None
                        async for event in self._traced_llm_call(
                            parent_run=root_run,
                            name=f"llm.{active_model}.direct_answer",
                            metadata={
                                "round": round_no,
                                "with_tools": False,
                                "reason": "no_tool_needed",
                            },
                            messages=effective_messages,
                            model=active_model,
                            tools=None,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            stop=stop,
                            seed=seed,
                        ):
                            if event.get("type") == "error" and event.get("status") == 429:
                                _da_429 = True
                                _da_event = event
                                break
                            if event.get("type") == "chunk" and event.get("text"):
                                full_response.append(event["text"])
                            yield event
                        if _da_429:
                            _da_ra = _da_event.get("retry_after") if _da_event else None
                            if _da_ra is not None and _da_ra <= 60:
                                log.warning("rate_limit_rpm_waiting", model=active_model, retry_after=_da_ra)
                                await asyncio.sleep(_da_ra + 1)
                                continue
                            if allow_fallback:
                                next_m = self._next_fallback(active_model)
                                if next_m:
                                    log.warning("rate_limit_fallback", from_model=active_model, to_model=next_m)
                                    active_model = next_m
                                    continue
                        _da_done = True
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

                    # Inject caller's user_id into rag_search so retrieval is
                    # scoped to that user's documents. The LLM never controls this.
                    if name == "rag_search" and user_id:
                        metadata_filter = args.get("metadata_filter")
                        if not isinstance(metadata_filter, dict):
                            metadata_filter = {}
                        metadata_filter["user_id"] = user_id
                        args["metadata_filter"] = metadata_filter

                    # ── Tool call span ──────────────────────────────────────────
                    tool_run = ls.create_run(
                        name=f"tool.{name}",
                        run_type="tool",
                        inputs={"name": name, "args": args},
                        parent_run=root_run,
                        metadata={"tool_call_id": tool_call_id, "round": round_no},
                        tags=[name],
                    )

                    result = await self.mcp.call_tool(name, args)

                    if name == "rag_search" and result.ok:
                        last_rag_result = result

                    # Build rich output for the tool span.
                    tool_output: dict = {
                        "success": result.ok,
                        "content_preview": (result.content or "")[:400],
                    }
                    if name == "rag_search":
                        # Parse chunk blocks so we can report count + previews.
                        blocks = (
                            [b.strip() for b in result.content.split("\n---\n") if b.strip()]
                            if result.ok and result.content
                            else []
                        )
                        tool_output["chunk_count"] = len(blocks)
                        if blocks:
                            tool_output["chunks_preview"] = [b[:200] for b in blocks[:3]]

                    ls.end_run(
                        tool_run,
                        outputs=tool_output,
                        error=None if result.ok else result.content,
                    )

                    effective_messages.append(
                        {
                            "role": "tool",
                            "name": name,
                            "tool_call_id": tool_call_id,
                            "content": result.content if result.ok else f"[{name} unavailable: {result.content}]",
                        }
                    )

                    any_tool_executed = True
                    total_tool_calls_executed += 1

                if any_tool_executed:
                    # Re-inject the last successful rag_search result as a system
                    # message so the model cannot overlook retrieved content.
                    # Only added when ok=True and content is not a "no results" response.
                    if (
                        last_rag_result is not None
                        and last_rag_result.ok
                        and "No relevant information" not in last_rag_result.content
                    ):
                        effective_messages.append({
                            "role": "system",
                            "content": (
                                "The following is the EXACT text retrieved from the user's documents. "
                                "You MUST answer using only the values and facts shown below. "
                                "Do not use your training knowledge to change, update, or contradict these values:\n\n"
                                + last_rag_result.content
                            ),
                        })

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

                    # ── LLM call 3: finalization pass (no tools) ───────────────
                    _fin_done = False
                    while not _fin_done:
                        _fin_429 = False
                        _fin_event: dict | None = None
                        async for event in self._traced_llm_call(
                            parent_run=root_run,
                            name=f"llm.{active_model}.finalization",
                            metadata={
                                "round": round_no,
                                "with_tools": False,
                                "reason": "finalization_after_tools",
                            },
                            messages=effective_messages,
                            model=active_model,
                            tools=None,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            stop=stop,
                            seed=seed,
                        ):
                            if event.get("type") == "error" and event.get("status") == 429:
                                _fin_429 = True
                                _fin_event = event
                                break
                            if event.get("type") == "chunk" and event.get("text"):
                                saw_final = True
                                full_response.append(event["text"])
                            if event.get("type") == "tool_call":
                                log.warning("tool_call_in_finalization_ignored")
                                continue
                            yield event
                        if _fin_429:
                            _fin_ra = _fin_event.get("retry_after") if _fin_event else None
                            if _fin_ra is not None and _fin_ra <= 60:
                                log.warning("rate_limit_rpm_waiting", model=active_model, retry_after=_fin_ra)
                                await asyncio.sleep(_fin_ra + 1)
                                continue
                            if allow_fallback:
                                next_m = self._next_fallback(active_model)
                                if next_m:
                                    log.warning("rate_limit_fallback", from_model=active_model, to_model=next_m)
                                    active_model = next_m
                                    continue
                        _fin_done = True

                    if not saw_final:
                        yield {
                            "type": "chunk",
                            "text": "Tool results received but the model did not produce a final response.",
                        }
                        yield {"type": "done", "finish_reason": "stop"}
                    break

            # ── Persist chat log ────────────────────────────────────────────────
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
                user_id=user_id,
            )
            await session.commit()

        finally:
            # End the root LangSmith trace regardless of how the generator exits
            # (normal completion, client disconnect, or exception).
            elapsed = round(time.monotonic() - start_time, 3)
            ls.end_run(
                root_run,
                outputs={
                    "response": "".join(full_response),
                    "total_seconds": elapsed,
                    "rounds": round_no,
                    "tool_calls_executed": total_tool_calls_executed,
                },
            )

    async def bulk_complete(
        self,
        *,
        session: AsyncSession,
        items: list[BulkChatItem],
    ) -> list[dict]:
        """
        Non-streaming bulk completions. Each item is an independent request
        processed sequentially. Returns a list of {model, response, error?} dicts.
        """
        results: list[dict] = []

        for idx, item in enumerate(items):
            model = item.model or settings.groq_default_model
            if model not in AVAILABLE_MODELS:
                results.append({"index": idx, "model": model, "response": None, "error": f"Unsupported model: {model}"})
                continue

            collected: list[str] = []
            error_msg: str | None = None

            try:
                async for event in self.client.stream_chat_completion(
                    messages=item.messages,
                    model=model,
                    tools=None,
                    temperature=item.temperature,
                ):
                    etype = event.get("type")
                    if etype == "chunk" and event.get("text"):
                        collected.append(event["text"])
                    elif etype == "error":
                        error_msg = event.get("message", "Unknown error")
                        break
                    elif etype == "done":
                        break
            except Exception as exc:
                error_msg = str(exc)

            results.append({
                "index": idx,
                "model": model,
                "response": "".join(collected) if collected else None,
                "error": error_msg,
            })

        return results
