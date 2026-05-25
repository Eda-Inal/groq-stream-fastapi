"""
Lightweight eval endpoint — returns only the routing decision (which tool the
model would call) without executing the tool or generating a final answer.

Used by eval/run_eval.py to test tool-routing accuracy at ~1/3 the token cost
of the full /chat/stream flow.
"""

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import AVAILABLE_MODELS, settings
from app.services.chat_service import ROUTING_SYSTEM_MESSAGE
from app.services.groq_client import LLMClient
from app.services.mcp.remote_client import RemoteMCPClient

router = APIRouter(prefix="/eval", tags=["eval"])

_llm = LLMClient()
_mcp = RemoteMCPClient()


class RouteRequest(BaseModel):
    question: str
    model: str | None = None


class RouteResponse(BaseModel):
    tool: str
    args: dict | None = None


@router.post("/route", response_model=RouteResponse)
async def route_question(payload: RouteRequest) -> RouteResponse:
    """
    Send a question through the routing LLM and return which tool it selects.
    The tool is never executed and no final answer is generated.
    """
    model = payload.model or settings.groq_default_model
    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    tools_schema = await _mcp.list_tools()
    messages = [
        ROUTING_SYSTEM_MESSAGE,
        {"role": "user", "content": payload.question},
    ]

    tool_state: dict[str, dict] = {}

    async for event in _llm.stream_chat_completion(
        messages=messages,
        model=model,
        tools=tools_schema,
    ):
        etype = event.get("type")
        if etype == "tool_call":
            for delta in event.get("tool_calls", []):
                key = str(delta.get("index", 0))
                if key not in tool_state:
                    tool_state[key] = {"name": None, "arguments": ""}
                fn = delta.get("function") or {}
                if fn.get("name"):
                    tool_state[key]["name"] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    tool_state[key]["arguments"] += fn["arguments"]
        elif etype in ("done", "error"):
            break

    if not tool_state:
        return RouteResponse(tool="none")

    first = next(iter(tool_state.values()))
    tool_name = first.get("name") or "none"
    raw_args = first.get("arguments", "")
    try:
        tool_args = json.loads(raw_args) if raw_args else None
    except Exception:
        tool_args = None

    return RouteResponse(tool=tool_name, args=tool_args)
