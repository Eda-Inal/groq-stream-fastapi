import httpx
import structlog
from app.services.tools.base import Tool
from app.core.config import settings

logger = structlog.get_logger()

class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Useful for answering questions about current events, news, or specific facts "
        "that require up-to-date information from the internet."
    )

    def __init__(self) -> None:
        self.api_key = settings.tavily_api_key
        self.base_url = "https://api.tavily.com/search"

    async def run(self, messages: list[dict[str, str]]) -> str:
        """
        Performs a real web search using the Tavily API.
        """
        query = messages[-1]["content"]
        
        if not self.api_key:
            logger.error("web_search_failed", reason="TAVILY_API_KEY_MISSING")
            return "Error: Web search is not configured (Missing API Key)."

        logger.info("web_search_executing", query=query)

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 3,
            "include_answer": False,
            "include_raw_content": False
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self.base_url, json=payload)
                
                if response.status_code != 200:
                    logger.error("tavily_api_error", status=response.status_code, body=response.text)
                    return f"Web search failed with status {response.status_code}."

                data = response.json()
                results = data.get("results", [])

                if not results:
                    return "Web search returned no results for this query."

                # Format the results into a clean string for the LLM
                formatted_results = []
                for res in results:
                    formatted_results.append(
                        f"Source: {res.get('url')}\nContent: {res.get('content')}"
                    )

                final_context = "\n\n---\n\n".join(formatted_results)
                return f"Web Search Results for '{query}':\n\n{final_context}"

        except httpx.HTTPError as e:
            logger.error("web_search_network_error", error=str(e))
            return f"Web search failed due to a network error: {str(e)}"
        except Exception as e:
            logger.error("web_search_unexpected_error", error=str(e))
            return f"An unexpected error occurred during web search: {str(e)}"