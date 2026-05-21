import httpx
from app.mcp_server.tools.base import Tool
from app.core.config import settings


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web. Call only when rag_search returns no relevant results."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"],
    }

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10)
        return self._client

    async def run(self, args: dict) -> str:
        """
        IMPORTANT RULE:
        - This method MUST NEVER raise an exception.
        - On any error, it must return a string.
        """
        try:
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return "Web search not used: missing or invalid query."

            if not getattr(settings, "tavily_api_key", None):
                return "Web search not available: missing Tavily API key."

            payload = {
                "api_key": settings.tavily_api_key,
                "query": query,
                "max_results": 3,
            }

            r = await self._get_client().post("https://api.tavily.com/search", json=payload)
            if r.status_code >= 400:
                return f"Web search failed: HTTP {r.status_code}"
            data = r.json()

            results = data.get("results", [])
            if not isinstance(results, list) or not results:
                return "Web search returned no results."

            return "\n".join(
                f"{i.get('url', '')}: {i.get('content', '')}"
                for i in results
                if isinstance(i, dict)
            ).strip() or "Web search returned results but could not format them."

        except Exception:
            return "Web search failed due to an unexpected error."
