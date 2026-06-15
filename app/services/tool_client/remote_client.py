from typing import List, Dict, Any

import httpx
import structlog

from app.core.config import settings
from app.tool_server.tools.base import ToolResult
from app.services.tool_client.client import ToolClient

logger = structlog.get_logger()


class RemoteToolClient(ToolClient):
    """
    HTTP-based client for remote tool discovery and execution.
    """

    def __init__(self) -> None:
        self.base_url = settings.tool_server_url
        self.timeout = settings.tool_server_timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def list_tools(self) -> List[Dict]:
        if not self.base_url:
            return []

        try:
            url = f"{self.base_url}/tools"
            r = await self._get_client().get(url)

            if r.status_code != 200:
                logger.warning("tool_client_list_tools_http_error", status=r.status_code)
                return []

            try:
                data = r.json()
            except Exception:
                logger.error("tool_client_list_tools_json_parse_error", exc_info=True)
                return []

            if isinstance(data, list):
                return data

            if isinstance(data, dict):
                tools = data.get("tools")
                if isinstance(tools, list):
                    return tools

            return []

        except Exception:
            logger.error("tool_client_list_tools_exception", exc_info=True)
            return []

    async def call_tool(self, name: str, args: dict, timeout: float | None = None) -> ToolResult:
        if not self.base_url:
            return ToolResult(ok=False, content="Tool server not configured.")

        log = logger.bind(tool=name)

        try:
            log.info("tool_client_call_started", arguments=args or {})

            url = f"{self.base_url}/tools/call"
            payload: Dict[str, Any] = {
                "name": name,
                "arguments": args or {},
            }

            request_kwargs: Dict[str, Any] = {"json": payload}
            if timeout is not None:
                request_kwargs["timeout"] = timeout

            r = await self._get_client().post(url, **request_kwargs)

            if r.status_code != 200:
                logger.error("tool_client_call_http_error", status=r.status_code)
                return ToolResult(ok=False, content=f"Tool call failed: HTTP {r.status_code}")

            try:
                data = r.json()
            except Exception:
                logger.error("tool_client_call_json_parse_error", exc_info=True)
                return ToolResult(ok=False, content="Tool call failed: invalid JSON response.")

            if not isinstance(data, dict):
                return ToolResult(ok=False, content="Tool call returned no result.")

            content = data.get("result")
            if not isinstance(content, str):
                content = str(content) if content is not None else "Tool call returned no result."

            ok = bool(data.get("success", True))
            return ToolResult(ok=ok, content=content)

        except Exception:
            logger.error("tool_client_call_exception", exc_info=True)
            return ToolResult(ok=False, content="Tool call failed due to unexpected error.")

    async def get_metrics(self) -> dict:
        if not self.base_url:
            return {}

        try:
            url = f"{self.base_url}/metrics"
            r = await self._get_client().get(url)

            if r.status_code != 200:
                logger.warning("tool_client_get_metrics_http_error", status=r.status_code)
                return {}

            try:
                data = r.json()
            except Exception:
                logger.error("tool_client_get_metrics_json_parse_error", exc_info=True)
                return {}

            return data if isinstance(data, dict) else {}

        except Exception:
            logger.error("tool_client_get_metrics_exception", exc_info=True)
            return {}
