# Copyright (C) 2026 Tonworio(Ton) Oguara
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Planner agent for travel planning and itinerary research."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from ..core.state import TravelPlannerState, get_task
from .prompts import travel_scout_instructions
from ..tools.web_search import web_search
from ..tools.deep_research import deep_research


def create_planner_agent(llm: ChatOpenAI):
    """Create a planner agent with travel planning capabilities.

    Args:
        llm: Chat model instance

    Returns:
        Tuple of (planner_agent_node, planner_tool_node) functions
    """
    # Use the @tool decorated functions directly with bind_tools
    planner_tools = [web_search, deep_research]
    planner_tools_by_name = {t.name: t for t in planner_tools}
    planner_model_with_tools = llm.bind_tools(planner_tools)

    def planner_agent_node(state: TravelPlannerState):
        """Planner agent with its own message history.

        Args:
            state: Current state with planner_messages history and the orchestrator's tasks

        Returns:
            Updates to state with new planner_messages and agent_results
        """
        # Work on this agent's sub-query from the orchestrator (fallback: full user query)
        task = get_task(state, "planner_agent")
        query = task.user_query if task else state.get("user_query", "")
        focus = task.focus if task and task.focus else "travel planning and itinerary"

        # Use planner-specific messages
        planner_msgs = state.get("planner_messages", [])

        # If this is first call, add the query (kept in history so later turns still see it)
        new_msgs = [] if planner_msgs else [HumanMessage(content=query)]

        messages = [SystemMessage(content=travel_scout_instructions)] + planner_msgs + new_msgs

        response = planner_model_with_tools.invoke(messages)

        return {
            "planner_messages": new_msgs + [response],
            "agent_results": [{
                "agent": "planner_agent",
                "focus": focus,
                "result": response.content
            }] if not response.tool_calls else []
        }

    def planner_tool_node(state: TravelPlannerState):
        """Execute planner tools based on agent tool calls.

        Args:
            state: Current state with planner_messages containing tool calls

        Returns:
            Updates to state with tool call results in planner_messages
        """
        planner_msgs = state.get("planner_messages", [])

        # Check if planner_messages exists and has content
        if not planner_msgs:
            return {"planner_messages": []}

        last_message = planner_msgs[-1]

        # Check if last_message has tool_calls
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return {"planner_messages": []}

        result = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            # Map tool name to the decorated function
            tool_func = planner_tools_by_name.get(tool_name)
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

        return {"planner_messages": result}

    return planner_agent_node, planner_tool_node
