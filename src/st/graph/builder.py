# Copyright (C) 2026 Tonworio(Ton) Oguara
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Graph builder for the multi-agent travel planning system."""

from typing import Literal, Sequence
from langgraph.types import Send
from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState, add_messages
from langgraph.prebuilt import ToolNode
from ..core.state import TravelPlannerState, AgentTask, ClassificationResult
from ..agents.orchestrator import classify_query_parallel
from ..agents.search_agent import create_search_agent
from ..agents.planner_agent import create_planner_agent
from ..synthesizer.synthesizer import create_synthesizer


def build_parallel_travel_agent(
    llm,
    tools: list = None,
    search_tools: list = None,
    planner_tools: list = None,
    checkpointer=None
) -> StateGraph:
    """Build a parallel multi-agent travel planning graph.

    This graph implements a map-reduce pattern:
    1. Orchestrator (classify_query_parallel) routes query to appropriate agents
    2. Parallel agents (search, planner) execute their tasks
    3. Synthesizer (deferred node) combines all results into the final answer once

    Args:
        llm: Base language model for agents
        tools:通用 tools for all agents
        search_tools: Tools for search agent (flight/hotel search)
        planner_tools: Tools for planner agent (web search, deep research)
        checkpointer: Optional LangGraph checkpointer (e.g. from st.memory.open_checkpointer)
            that persists each thread's state between runs for multi-turn memory

    Returns:
        Compiled StateGraph ready for use
    """
    # Default tools if not provided
    if tools is None:
        tools = []
    if search_tools is None:
        search_tools = []
    if planner_tools is None:
        planner_tools = []

    # Create tool instances if not provided (using @tool decorated functions)
    if not search_tools:
        from ..tools.flight_search import search_flights
        from ..tools.hotel_search import search_hotels
        from ..tools.web_search import web_search
        from ..tools.deep_research import deep_research
        search_tools = [search_flights, search_hotels]
        planner_tools = [web_search, deep_research]

    # Create agent components (no longer pass tool instances)
    search_node, search_tool_node = create_search_agent(llm)
    planner_node, planner_tool_node = create_planner_agent(llm)
    synthesizer_node = create_synthesizer(llm)

    # Build the workflow graph
    workflow = StateGraph(TravelPlannerState)

    # Add nodes
    workflow.add_node("orchestrator", classify_query_parallel)
    workflow.add_node("search_agent", search_node)
    workflow.add_node("search_tool", search_tool_node)
    workflow.add_node("planner_agent", planner_node)
    workflow.add_node("planner_tool", planner_tool_node)
    # Deferred: runs once, after every dispatched agent lane has finished (fan-in)
    workflow.add_node("synthesizer", synthesizer_node, defer=True)

    # Define conditional edges for orchestrator
    def route_to_agent(state: TravelPlannerState) -> Sequence[Literal["search_agent", "planner_agent"]]:
        """Route query to appropriate agents based on classification."""
        # Use tasks from orchestrator to determine which agents to invoke
        tasks = state.get("tasks", [])

        if not tasks:
            return ["search_agent", "planner_agent"]  # Default to both

        agents_to_invoke = set()
        for task in tasks:
            if task.source == "search_agent":
                agents_to_invoke.add("search_agent")
            elif task.source == "planner_agent":
                agents_to_invoke.add("planner_agent")

        return list(agents_to_invoke) if agents_to_invoke else ["search_agent", "planner_agent"]

    workflow.add_conditional_edges(
        "orchestrator",
        route_to_agent,
        path_map=["search_agent", "planner_agent"]
    )

    # Tool nodes route back to their agent after tool execution
    workflow.add_edge("search_tool", "search_agent")
    workflow.add_edge("planner_tool", "planner_agent")

    # Agent conditional edges: loop through the tool node while the agent makes tool calls,
    # otherwise hand over to the synthesizer
    def route_from_search_agent(state: TravelPlannerState) -> Literal["search_tool", "synthesizer"]:
        """Route search agent to its tool node or to the synthesizer."""
        search_msgs = state.get("search_messages", [])
        if search_msgs and getattr(search_msgs[-1], "tool_calls", None):
            return "search_tool"
        return "synthesizer"

    def route_from_planner_agent(state: TravelPlannerState) -> Literal["planner_tool", "synthesizer"]:
        """Route planner agent to its tool node or to the synthesizer."""
        planner_msgs = state.get("planner_messages", [])
        if planner_msgs and getattr(planner_msgs[-1], "tool_calls", None):
            return "planner_tool"
        return "synthesizer"

    workflow.add_conditional_edges(
        "search_agent",
        route_from_search_agent
    )
    workflow.add_conditional_edges(
        "planner_agent",
        route_from_planner_agent
    )

    workflow.add_edge("synthesizer", END)

    # Set entry point
    workflow.set_entry_point("orchestrator")

    # Compile the graph (with a checkpointer, state is saved per thread after every step)
    graph = workflow.compile(checkpointer=checkpointer)

    return graph
