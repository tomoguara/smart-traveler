"""Deep research tool backed by a deepagents itinerary-research sub-agent.

The planner (itinerary) agent invokes this tool for deep analysis such as
multi-day itineraries and in-depth destination research. Internally it runs a
deep agent (planning + Tavily web research) and returns its final insights.
"""

import os
from functools import lru_cache

from langchain_core.tools import tool

from ..agents.prompts import research_instructions

# Default model for the research sub-agent; override with DEEP_RESEARCH_MODEL.
DEFAULT_DEEP_RESEARCH_MODEL = "gpt-4o-mini"

# Caps the sub-agent's LangGraph steps (deepagents defaults to 9,999).
DEEP_RESEARCH_RECURSION_LIMIT = 60


@lru_cache(maxsize=1)
def get_itinerary_research_agent():
    """Build (once) the deepagents itinerary-research sub-agent.

    Returns:
        Compiled deep agent graph using Tavily search as its research tool
    """
    try:
        from deepagents import create_deep_agent
    except ImportError:
        raise ImportError("Please install deepagents with `uv add deepagents`")

    from langchain_openai import ChatOpenAI
    from langchain_tavily import TavilySearch

    model = ChatOpenAI(
        model=os.getenv("DEEP_RESEARCH_MODEL", DEFAULT_DEEP_RESEARCH_MODEL),
        temperature=0.2,
    )

    internet_search = TavilySearch(
        max_results=5,
        topic="general",
        search_depth="advanced",
        include_answer=True,
    )

    return create_deep_agent(
        model=model,
        tools=[internet_search],
        system_prompt=research_instructions,
        name="itinerary_research_agent",
    )


@tool(
    "deep_research",
    description=(
        "Run deep, multi-step travel research with a dedicated research sub-agent. "
        "Use for day-by-day itineraries, multi-city route plans, and in-depth "
        "destination analysis. Returns a researched itinerary/insights with rationale."
    ),
)
def deep_research(query: str) -> str:
    """Perform deep travel research and itinerary planning.

    Args:
        query: Self-contained research request including destination, trip length,
            traveller interests, budget and any other known constraints

    Returns:
        Researched itinerary and insights as markdown text
    """
    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError("TAVILY_API_KEY environment variable not set")

    try:
        agent = get_itinerary_research_agent()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": DEEP_RESEARCH_RECURSION_LIMIT},
        )
        return result["messages"][-1].text
    except Exception as e:
        return f"Error during deep research: {str(e)}"
