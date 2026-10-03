# Copyright (C) 2026 Tonworio(Ton) Oguara
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Search agent for flight and hotel searches."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from ..core.state import TravelPlannerState, get_task
from .prompts import flightHotelSearch_instructions
from ..tools.flight_search import search_flights
from ..tools.hotel_search import search_hotels


def create_search_agent(llm: ChatOpenAI):
    """Create a search agent with flight and hotel search capabilities.

    Args:
        llm: Chat model instance

    Returns:
        Tuple of (search_agent_node, search_tool_node) functions
    """
    # Use the @tool decorated functions directly with bind_tools
    search_tools = [search_flights, search_hotels]
    search_tools_by_name = {t.name: t for t in search_tools}
    search_model_with_tools = llm.bind_tools(search_tools)

    def search_agent_node(state: TravelPlannerState):
        """Search agent with its own message history.

        Args:
            state: Current state with search_messages history and the orchestrator's tasks

        Returns:
            Updates to state with new search_messages and agent_results
        """
        # Work on this agent's sub-query from the orchestrator (fallback: full user query)
        task = get_task(state, "search_agent")
        query = task.user_query if task else state.get("user_query", "")
        focus = task.focus if task and task.focus else "flights and hotels"

        # Use search-specific messages
        search_msgs = state.get("search_messages", [])

        # If this is first call, add the query (kept in history so later turns still see it)
        new_msgs = [] if search_msgs else [HumanMessage(content=query)]

        messages = [SystemMessage(content=flightHotelSearch_instructions)] + search_msgs + new_msgs

        response = search_model_with_tools.invoke(messages)

        return {
            "search_messages": new_msgs + [response],
            "agent_results": [{
                "agent": "search_agent",
                "focus": focus,
                "result": response.content
            }] if not response.tool_calls else []
        }

    def search_tool_node(state: TravelPlannerState):
        """Execute search tools based on agent tool calls.

        Args:
            state: Current state with search_messages containing tool calls

        Returns:
            Updates to state with tool call results in search_messages
        """
        search_msgs = state.get("search_messages", [])

        # Check if search_messages exists and has content
        if not search_msgs:
            return {"search_messages": []}

        last_message = search_msgs[-1]

        # Check if last_message has tool_calls
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return {"search_messages": []}

        result = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            # Map tool name to the decorated function
            tool_func = search_tools_by_name.get(tool_name)
            if tool_func is None:
                # Every tool call needs a ToolMessage reply, or the next LLM call fails
                result.append(ToolMessage(content=f"Error: unknown tool {tool_name}", tool_call_id=tool_call["id"]))
                continue

            # Execute the tool function directly
            try:
                observation = tool_func.invoke(tool_call["args"])
                result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
            except Exception as e:
                result.append(ToolMessage(content=f"Error executing tool {tool_name}: {str(e)}", tool_call_id=tool_call["id"]))

        return {"search_messages": result}

    return search_agent_node, search_tool_node
