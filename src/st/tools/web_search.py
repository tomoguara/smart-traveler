# Copyright (C) 2026 Tonworio(Ton) Oguara
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Web search tool using Tavily."""

import json
from typing import Literal, Optional
from langchain_core.tools import tool

# Cap on characters kept per result snippet (keeps the LLM context small)
MAX_SNIPPET_CHARS = 1500


def summarize_web_results(result: dict) -> dict:
    """Keep Tavily's answer plus title/url/snippet for each result."""
    return {
        "answer": result.get("answer"),
        "results": [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": (r.get("content") or "")[:MAX_SNIPPET_CHARS],
            }
            for r in result.get("results", [])
        ],
    }


@tool(
    "web_search",
    description="Search the web for travel information using Tavily Search API",
    return_direct=True
)
def web_search(
    query: str,
    search_depth: Literal["basic", "advanced"] = "basic",
    topic: Literal["general", "news", "finance"] = "general",
    time_range: Optional[Literal["day", "week", "month", "year"]] = None,
    max_results: Optional[int] = None,
) -> str:
    """Search the web for travel information.

    Args:
        query: Search query
        search_depth: Search depth: 'basic' or 'advanced'
        topic: Topic category: 'general', 'news' or 'finance'
        time_range: Only return results from the last 'day', 'week', 'month' or 'year'
        max_results: Maximum number of results to return

    Returns:
        Compact JSON with Tavily's answer and result snippets
    """
    import os
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set")

    # Import here to avoid issues if package not installed
    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        return "Error: langchain-tavily package not installed. Please install with: uv pip install langchain-tavily"

    # Create Tavily search tool (snippets only; raw page content floods the context)
    search_tool = TavilySearch(
        max_results=max_results or 5,
        include_answer=True,
        include_raw_content=False,
    )

    search_args = {"query": query, "search_depth": search_depth, "topic": topic}
    if time_range:
        search_args["time_range"] = time_range

    try:
        result = search_tool.invoke(search_args)
        if not isinstance(result, dict):
            return str(result)
        if "error" in result:
            return f"Error searching web: {result['error']}"
        return json.dumps(summarize_web_results(result), ensure_ascii=False)
    except Exception as e:
        return f"Error searching web: {str(e)}"
