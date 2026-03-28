"""
Web search tool — search the web using Tavily API.
"""

from dataclasses import dataclass

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


@dataclass
class WebSearchTool:
    """
    Search the web using Tavily API.

    Attributes:
        api_key: Tavily API key (optional, will use env TAVILY_API_KEY if not provided)
    """

    api_key: str = None

    def run(self, query: str, max_results: int = 5) -> str:
        """
        Search the web.

        Args:
            query: Search query
            max_results: Maximum number of results (default 5)

        Returns:
            Formatted search results or error message
        """
        if not TAVILY_AVAILABLE:
            return "[Error: tavily package not installed. Run: pip install tavily]"

        import os
        api_key = self.api_key or os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "[Error: TAVILY_API_KEY not set. Provide api_key or set TAVILY_API_KEY env var]"

        try:
            client = TavilyClient(api_key=api_key)
            results = client.search(query=query, max_results=max_results)

            if not results.get('results'):
                return "[No results found]"

            lines = []
            for i, item in enumerate(results['results'], 1):
                title = item.get('title', 'No title')
                url = item.get('url', '')
                snippet = item.get('content', '')[:200]
                lines.append(f"{i}. {title}\n   {url}\n   {snippet}\n")

            return "\n".join(lines)

        except Exception as e:
            return f"[Error: Web search failed: {e}]"


# Decorator-based tool
def web_search_tool(query: str, max_results: int = 5) -> str:
    """Search the web. Args: query (str), max_results (int, default 5)."""
    tool = WebSearchTool()
    return tool.run(query, max_results)
