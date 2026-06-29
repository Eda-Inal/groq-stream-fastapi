from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import structlog

from app.core.config import AVAILABLE_MODELS
from app.services.groq_client import LLMClient
from app.services.tool_client.remote_client import RemoteToolClient
from app.utils.token_counter import (
    count_tokens,
    estimate_messages_tokens,
    truncate_text_to_token_budget,
)

logger = structlog.get_logger()

_MIN_SEARCHES = 3
_MAX_SEARCHES = 5
_ROUTING_MAX_TOKENS = 256
_MAX_CONTENT_TOKENS = 150        # per individual Tavily result content
_FINALIZATION_INPUT_MAX = 800    # collected results content budget (per-result truncation)
_FINALIZATION_OVERHEAD = 900     # system prompt + URL lines + separators + message format
_FINALIZATION_OUTPUT_MAX = 1000  # report output token cap
_DEFAULT_TPM = 8000              # fallback for models without rate_limits in config


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _routing_system_prompt(topic: str) -> str:
    return (
        f"You are a research agent investigating '{topic}'.\n\n"
        "Each round you will see the results of your previous searches. "
        "Read them carefully and decide what to search for next based on what you found:\n"
        "- What was mentioned but not explained?\n"
        "- What claim needs verification or more detail?\n"
        "- What important angle is still missing?\n\n"
        "Rules:\n"
        "- Each query must follow from what you actually read in the previous results. "
        "Do not follow a fixed list of angles — let the findings guide you.\n"
        "- Do NOT write any text response — only make tool calls.\n"
        "- When you have enough information to write a complete report, stop calling tools."
    )


def _finalization_system_prompt(topic: str, language: str) -> str:
    lang_label = "English" if language == "en" else "Turkish"

    if language == "en":
        structure = (
            "## Summary\n2-3 sentences covering the core finding.\n\n"
            "## Key Findings\n3-5 bullet points with the most important facts.\n\n"
            "## Detailed Analysis\n3-4 paragraphs covering: overview, current state, "
            "key examples, challenges and outlook.\n\n"
            "## Sources\nNumbered list of all URLs from the search results."
        )
    else:
        structure = (
            "## Özet\nTemel bulguyu özetleyen 2-3 cümle.\n\n"
            "## Ana Bulgular\nEn önemli 3-5 madde.\n\n"
            "## Detaylı Analiz\nGenel bakış, mevcut durum, örnekler ve "
            "zorluklar/gelecek perspektifini kapsayan 3-4 paragraf.\n\n"
            "## Kaynaklar\nArama sonuçlarındaki tüm URL'lerin numaralı listesi."
        )

    return (
        f"Based on the web search results provided, write a comprehensive research "
        f"report about '{topic}' in {lang_label}.\n\n"
        "Use ONLY the information from the search results. Do not add external knowledge.\n\n"
        f"Report structure (markdown):\n{structure}"
    )


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

def _compute_budgets(model: str) -> dict:
    model_info = AVAILABLE_MODELS.get(model, {})
    tpm = model_info.get("rate_limits", {}).get("tpm", _DEFAULT_TPM)
    # Finalization uses fixed caps — the early break check enforces the TPM limit
    # dynamically using actual message sizes, so these don't need to be derived
    # from the remaining budget.
    return {
        "tpm": tpm,
        "finalization_input_max": _FINALIZATION_INPUT_MAX,
        "finalization_output_max": _FINALIZATION_OUTPUT_MAX,
    }


# ---------------------------------------------------------------------------
# Result formatting (structure-aware — URL is never truncated)
# ---------------------------------------------------------------------------

def _parse_tavily_results(content: str) -> list[dict]:
    """
    Parse web_search tool output into {url, content} dicts.
    Each line is formatted as "url: content" by web_search.py.
    """
    results = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if ": " in line:
            url, text = line.split(": ", 1)
            results.append({"url": url.strip(), "content": text.strip()})
        else:
            results.append({"url": "", "content": line})
    return results


def _format_result(round_num: int, query: str, url: str, content: str, max_tokens: int) -> str:
    truncated = truncate_text_to_token_budget(content, max_tokens)
    url_line = f"Source: {url}" if url else ""
    parts = [f'[Round {round_num} — Query: "{query}"]']
    if url_line:
        parts.append(url_line)
    parts.append(truncated)
    return "\n".join(p for p in parts if p).strip()


def _build_finalization_messages(
    topic: str,
    language: str,
    collected_results: list[dict],
    input_budget: int,
) -> list[dict]:
    """
    Build a 2-message list for finalization.
    Routing history is intentionally NOT carried over to keep the context clean
    and the token count predictable. Budget is distributed evenly across rounds
    so later (more specific) searches are never disproportionately cut.
    """
    if not collected_results:
        results_block = "No search results were collected."
    else:
        rounds: dict[int, list[dict]] = {}
        for r in collected_results:
            rounds.setdefault(r["round"], []).append(r)

        n_rounds = len(rounds)
        per_round_budget = max(100, input_budget // n_rounds) if n_rounds else input_budget

        blocks: list[str] = []
        for round_num in sorted(rounds.keys()):
            items = rounds[round_num]
            per_result_budget = max(50, per_round_budget // len(items))
            for item in items:
                blocks.append(
                    _format_result(
                        round_num,
                        item["query"],
                        item["url"],
                        item["content"],
                        per_result_budget,
                    )
                )
        results_block = "\n\n---\n\n".join(blocks)

    return [
        {"role": "system", "content": _finalization_system_prompt(topic, language)},
        {
            "role": "user",
            "content": f"Research topic: {topic}\n\nSearch results:\n\n{results_block}",
        },
    ]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ResearchService:
    def __init__(self) -> None:
        self.llm_client = LLMClient()
        self.tool_client = RemoteToolClient()

    async def _get_web_search_tool(self) -> dict:
        tools = await self.tool_client.list_tools()
        for t in tools:
            if isinstance(t, dict) and t.get("function", {}).get("name") == "web_search":
                return t
        raise RuntimeError("web_search tool not available from tool-server")

    async def stream_research(
        self,
        *,
        topic: str,
        model: str,
        user_id: str | None,
        language: str,
    ) -> AsyncIterator[dict]:
        log = logger.bind(model=model, user_id=user_id)
        start_time = time.monotonic()

        budgets = _compute_budgets(model)

        log.info(
            "research_loop_start",
            topic=topic,
            model=model,
            tpm_budget=budgets["tpm"],
            finalization_input_max=budgets["finalization_input_max"],
            finalization_output_max=budgets["finalization_output_max"],
            min_searches=_MIN_SEARCHES,
            max_searches=_MAX_SEARCHES,
        )

        # Tool discovery
        try:
            web_search_tool = await self._get_web_search_tool()
        except RuntimeError as exc:
            log.error("research_stream_error", phase="discovery", error=str(exc))
            yield {"type": "error", "message": str(exc), "phase": "discovery"}
            return

        yield {"type": "start", "topic": topic, "model": model, "tpm_budget": budgets["tpm"]}

        messages: list[dict] = [
            {"role": "system", "content": _routing_system_prompt(topic)},
            {"role": "user", "content": topic},
        ]

        # Separate accumulator for finalization — raw content, un-truncated
        collected_results: list[dict] = []
        search_count = 0
        round_num = 0
        estimated_tokens_used = 0
        exit_reason = "model_stopped"

        try:
            while round_num < _MAX_SEARCHES:
                round_num += 1

                # Stop early if this routing call + finalization would exceed TPM budget.
                # Uses actual message size (not a flat estimate) so the check is accurate
                # as messages grow each round.
                current_input = estimate_messages_tokens(messages)
                finalization_cost = (
                    budgets["finalization_input_max"]
                    + _FINALIZATION_OVERHEAD
                    + budgets["finalization_output_max"]
                )
                projected_total = (
                    estimated_tokens_used + current_input + _ROUTING_MAX_TOKENS + finalization_cost
                )
                if projected_total > budgets["tpm"]:
                    log.warning(
                        "research_loop_early_break",
                        reason="tpm_budget_low",
                        projected_total=projected_total,
                        tpm_budget=budgets["tpm"],
                        search_count=search_count,
                    )
                    exit_reason = "tpm_budget_low"
                    break

                estimated_tokens_used += current_input + _ROUTING_MAX_TOKENS

                log.info(
                    "research_routing_call",
                    round=round_num,
                    message_count=len(messages),
                    estimated_input_tokens=current_input,
                    estimated_tokens_used_total=estimated_tokens_used,
                    estimated_remaining=budgets["tpm"] - estimated_tokens_used,
                )

                # Routing LLM call — temperature=0, tools only, no text output expected
                tool_calls_raw: list[dict] = []
                had_error = False

                async for event in self.llm_client.stream_chat_completion(
                    messages=messages,
                    model=model,
                    tools=[web_search_tool],
                    stream=False,
                    temperature=0,
                    max_tokens=_ROUTING_MAX_TOKENS,
                    call_type="research_routing",
                ):
                    etype = event.get("type")
                    if etype == "tool_call":
                        tool_calls_raw.extend(event.get("tool_calls", []))
                    elif etype == "error":
                        log.error(
                            "research_stream_error",
                            phase="loop",
                            error=event.get("message", ""),
                        )
                        yield {
                            "type": "error",
                            "message": event.get("message", "LLM error during research loop"),
                            "phase": "loop",
                        }
                        had_error = True
                        break
                    elif etype == "done":
                        break

                if had_error:
                    return

                if not tool_calls_raw:
                    if search_count >= _MIN_SEARCHES:
                        break  # model decided it has enough; respect that

                    # Not enough searches yet — nudge the model to continue
                    log.info(
                        "research_nudge_injected",
                        round=round_num,
                        search_count=search_count,
                        reason="min_searches_not_met",
                    )
                    yield {"type": "nudge", "round": round_num, "search_count": search_count}
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You have only performed {search_count} search(es). "
                            f"Please search at least {_MIN_SEARCHES} different angles "
                            "before concluding."
                        ),
                    })
                    continue

                # Build assistant message with tool_calls in OpenAI format
                openai_tool_calls = []
                for i, tc in enumerate(tool_calls_raw):
                    fn = tc.get("function") or {}
                    call_id = tc.get("id") or f"call_{round_num}_{i}"
                    openai_tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fn.get("name", "web_search"),
                            "arguments": fn.get("arguments", "{}"),
                        },
                    })

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": openai_tool_calls,
                })

                # Execute each tool call and append results
                for tc in openai_tool_calls:
                    call_id = tc["id"]
                    fn = tc["function"]
                    tool_name = fn.get("name", "web_search")
                    raw_args = fn.get("arguments", "{}")

                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {}

                    query = args.get("query", topic)

                    log.info(
                        "research_web_search_called",
                        round=round_num,
                        query=query,
                        search_count_after=search_count + 1,
                    )
                    yield {"type": "searching", "query": query, "round": round_num}

                    t_call = time.monotonic()
                    result = await self.tool_client.call_tool(tool_name, args)
                    elapsed_ms = round((time.monotonic() - t_call) * 1000, 1)

                    if result.ok and result.content.strip():
                        search_count += 1
                        raw_tokens = count_tokens(result.content)
                        parsed = _parse_tavily_results(result.content)

                        # Store raw content for finalization budget allocation
                        for item in parsed:
                            collected_results.append({
                                "round": round_num,
                                "query": query,
                                "url": item["url"],
                                "content": item["content"],
                            })

                        # Truncated content for the routing messages list
                        formatted_parts = [
                            _format_result(
                                round_num, query,
                                item["url"], item["content"],
                                _MAX_CONTENT_TOKENS,
                            )
                            for item in parsed
                        ]
                        tool_result_content = "\n\n".join(formatted_parts)
                        truncated_tokens = count_tokens(tool_result_content)

                        log.info(
                            "research_web_search_completed",
                            round=round_num,
                            query=query,
                            ok=True,
                            raw_content_tokens=raw_tokens,
                            truncated_content_tokens=truncated_tokens,
                            elapsed_ms=elapsed_ms,
                        )

                        yield {
                            "type": "found",
                            "query": query,
                            "round": round_num,
                            "source_count": len(parsed),
                            "preview": result.content[:120].replace("\n", " "),
                        }
                    else:
                        # Failed search — do not increment search_count
                        tool_result_content = "Search returned no results."
                        log.warning(
                            "research_web_search_completed",
                            round=round_num,
                            query=query,
                            ok=False,
                            raw_content_tokens=0,
                            truncated_content_tokens=0,
                            elapsed_ms=elapsed_ms,
                        )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": tool_result_content,
                    })

            if round_num >= _MAX_SEARCHES and exit_reason == "model_stopped":
                exit_reason = "max_searches_reached"

        except Exception as exc:
            log.error("research_stream_error", phase="loop", error=str(exc), exc_info=True)
            yield {
                "type": "error",
                "message": "Unexpected error during research loop.",
                "phase": "loop",
            }
            return

        log.info(
            "research_loop_complete",
            total_rounds=round_num,
            total_searches=search_count,
            exit_reason=exit_reason,
            elapsed_ms=round((time.monotonic() - start_time) * 1000, 1),
        )

        # Finalization — clean 2-message context, no routing history
        yield {"type": "finalizing", "total_searches": search_count}

        fin_messages = _build_finalization_messages(
            topic=topic,
            language=language,
            collected_results=collected_results,
            input_budget=budgets["finalization_input_max"],
        )

        raw_results_tokens = sum(count_tokens(r["content"]) for r in collected_results)
        fin_input_tokens = estimate_messages_tokens(fin_messages)

        log.info(
            "research_finalization_start",
            total_searches=search_count,
            collected_results_count=len(collected_results),
            raw_results_tokens=raw_results_tokens,
            after_budget_tokens=fin_input_tokens,
            finalization_message_count=len(fin_messages),
        )

        try:
            t_fin = time.monotonic()
            report_chars = 0

            async for event in self.llm_client.stream_chat_completion(
                messages=fin_messages,
                model=model,
                tools=None,
                stream=True,
                temperature=0.7,
                max_tokens=budgets["finalization_output_max"],
                call_type="research_finalization",
            ):
                etype = event.get("type")
                if etype == "chunk" and event.get("text"):
                    text = event["text"]
                    report_chars += len(text)
                    yield {"type": "report_chunk", "text": text}
                elif etype == "error":
                    log.error(
                        "research_stream_error",
                        phase="finalization",
                        error=event.get("message", ""),
                    )
                    yield {
                        "type": "error",
                        "message": event.get("message", "Finalization error"),
                        "phase": "finalization",
                    }
                    return
                elif etype == "done":
                    break

            log.info(
                "research_finalization_complete",
                elapsed_ms=round((time.monotonic() - t_fin) * 1000, 1),
                report_chars=report_chars,
                estimated_output_tokens=report_chars // 4,
            )

        except Exception as exc:
            log.error("research_stream_error", phase="finalization", error=str(exc), exc_info=True)
            yield {
                "type": "error",
                "message": "Unexpected error during finalization.",
                "phase": "finalization",
            }
            return

        total_elapsed = round(time.monotonic() - start_time, 1)

        log.info(
            "research_stream_done",
            total_elapsed_ms=round((time.monotonic() - start_time) * 1000, 1),
            total_searches=search_count,
            model=model,
            user_id=user_id,
        )

        yield {"type": "done", "total_searches": search_count, "elapsed_seconds": total_elapsed}
