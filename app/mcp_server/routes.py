from fastapi import APIRouter
from pydantic import BaseModel
import structlog

from app.mcp_server.tools.registry import ToolRegistry
from app.services.rag_metrics import rag_metrics

logger = structlog.get_logger()

router = APIRouter()
registry = ToolRegistry()


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict


@router.get("/tools")
async def list_tools():
    tools = registry.openai_tools()
    return {"tools": tools}


@router.get("/metrics")
async def get_metrics() -> dict:
    """Expose in-process RAG metrics so the API can proxy them."""
    return rag_metrics.snapshot()


@router.post("/tools/call")
async def call_tool(payload: ToolCallRequest):
    name = payload.name
    args = payload.arguments or {}

    log = logger.bind(tool=name)
    log.info("tool_call_started", name=name)

    tool = registry.get(name)
    if tool is None:
        return {"result": "Tool not found"}

    try:
        result = await tool.run(args)
        if not isinstance(result, str):
            result = str(result)
        return {"result": result}
    except Exception:
        log.error("tool_call_failed", exc_info=True)
        return {"result": "Tool execution failed"}
