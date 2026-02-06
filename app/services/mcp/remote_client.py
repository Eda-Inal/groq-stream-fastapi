from typing import List, Dict, Any

import httpx
import structlog

from app.core.config import settings
from app.services.mcp.client import MCPClient

logger = structlog.get_logger()


class RemoteMCPClient(MCPClient):
    """
    HTTP-based MCP client for remote tool discovery and execution.
    """

    def __init__(self) -> None:
        self.base_url = settings.mcp_server_url
        self.timeout = settings.mcp_timeout

    async def list_tools(self) -> List[Dict]:
        if not self.base_url:
            return []

        try:
            url = f"{self.base_url}/tools"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(url)

            if r.status_code != 200:
                logger.warning("mcp_remote_list_tools_http_error", status=r.status_code)
                return []

            try:
                data = r.json()
            except Exception:
                logger.error("mcp_remote_list_tools_json_parse_error", exc_info=True)
                return []

            if isinstance(data, list):
                return data

            if isinstance(data, dict):
                tools = data.get("tools")
                if isinstance(tools, list):
                    return tools

            return []

        except Exception:
            logger.error("mcp_remote_list_tools_exception", exc_info=True)
            return []

    async def call_tool(self, name: str, args: dict) -> str:
        if not self.base_url:
            return "MCP server not configured."

        log = logger.bind(tool=name)

        try:
            log.info("mcp_remote_call_started")

            url = f"{self.base_url}/tools/call"
            payload: Dict[str, Any] = {
                "name": name,
                "arguments": args or {},
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(url, json=payload)

            if r.status_code != 200:
                logger.error(
                    "mcp_remote_call_http_error",
                    status=r.status_code,
                )
                return f"MCP call failed: HTTP {r.status_code}"

            try:
                data = r.json()
            except Exception:
                logger.error("mcp_remote_call_json_parse_error", exc_info=True)
                return "MCP call failed: invalid JSON response."

            result = data.get("result") if isinstance(data, dict) else None

            if isinstance(result, str):
                return result

            if result is None:
                return "MCP call returned no result."

            return str(result)

        except Exception:
            logger.error("mcp_remote_call_exception", exc_info=True)
            return "MCP call failed due to unexpected error."
