# Copyright (C) 2026 Tonworio(Ton) Oguara
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the planner (itinerary) agent's use of deep_research (no network calls)."""

from langchain_core.messages import AIMessage, ToolMessage

import st.agents.orchestrator as orchestrator_module
from st.agents import create_planner_agent
from st.agents.prompts import travel_scout_instructions
from st.core import AgentTask, ClassificationResult
from st.graph import build_parallel_travel_agent

from conftest import StubLLM


def _tool_call(name, args, call_id="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_planner_binds_web_search_and_deep_research():
    llm = StubLLM()

    create_planner_agent(llm)

    assert llm.bound_tools == ["web_search", "deep_research"]


def test_prompt_references_only_bound_tools():
    assert "web_search" in travel_scout_instructions
    assert "deep_research" in travel_scout_instructions
    assert "internet_search" not in travel_scout_instructions
    assert "call_itinerary_research_agent" not in travel_scout_instructions


def test_tool_node_runs_deep_research(stub_research_agent):
    agent = stub_research_agent(reply="Day 1: Sintra day trip")
    _, planner_tool_node = create_planner_agent(StubLLM())

    update = planner_tool_node({"planner_messages": [_tool_call("deep_research", {"query": "Lisbon 3 days"})]})

    [message] = update["planner_messages"]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call-1"
    assert message.content == "Day 1: Sintra day trip"
    assert agent.calls[0][0]["messages"][0]["content"] == "Lisbon 3 days"


def test_tool_node_reports_tool_exceptions(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    _, planner_tool_node = create_planner_agent(StubLLM())

    update = planner_tool_node({"planner_messages": [_tool_call("deep_research", {"query": "Lisbon"})]})

    [message] = update["planner_messages"]
    assert message.content.startswith("Error executing tool deep_research: TAVILY_API_KEY")


def test_tool_node_answers_unknown_tool_calls():
    _, planner_tool_node = create_planner_agent(StubLLM())

    update = planner_tool_node({"planner_messages": [_tool_call("book_hotel", {}, call_id="call-9")]})

    [message] = update["planner_messages"]
    assert message.tool_call_id == "call-9"
    assert message.content == "Error: unknown tool book_hotel"


class _StubOrchestratorLLM:
    """Replaces the ChatOpenAI the orchestrator builds; routes to the planner only."""

    def __init__(self, **kwargs):
        pass

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return ClassificationResult(tasks=[AgentTask(
            source="planner_agent", user_query="Plan a 3-day food itinerary for Lisbon", focus="food itinerary")])


def test_graph_planner_uses_deep_research_then_synthesizes(monkeypatch, stub_research_agent):
    stub_research_agent(reply="Day 1: Alfama food tour")
    monkeypatch.setattr(orchestrator_module, "ChatOpenAI", _StubOrchestratorLLM)
    llm = StubLLM([
        _tool_call("deep_research", {"query": "3-day Lisbon food itinerary"}),  # planner turn 1
        AIMessage(content="ITINERARY WITH RATIONALE"),                           # planner turn 2
        AIMessage(content="FINAL RECOMMENDATION"),                               # synthesizer
    ])
    graph = build_parallel_travel_agent(llm)

    result = graph.invoke({
        "messages": [],
        "user_query": "Plan a 3-day food trip to Lisbon",
        "tasks": [],
        "requires_synthesis": False,
        "agent_results": [],
        "final_answer": "",
        "search_messages": [],
        "planner_messages": [],
    })

    assert result["final_answer"] == "FINAL RECOMMENDATION"
    assert result["agent_results"] == [{
        "agent": "planner_agent",
        "focus": "food itinerary",
        "result": "ITINERARY WITH RATIONALE",
    }]
    # The planner worked on its sub-query, and it stays in the history for later turns
    first = result["planner_messages"][0]
    assert (first.type, first.content) == ("human", "Plan a 3-day food itinerary for Lisbon")
    later_turn_query = llm.seen[1][1]
    assert (later_turn_query.type, later_turn_query.content) == ("human", "Plan a 3-day food itinerary for Lisbon")
    tool_messages = [m for m in result["planner_messages"] if isinstance(m, ToolMessage)]
    assert [m.content for m in tool_messages] == ["Day 1: Alfama food tour"]
    # The planner's second turn saw the research insights
    assert any(isinstance(m, ToolMessage) and m.content == "Day 1: Alfama food tour" for m in llm.seen[1])
    assert result["search_messages"] == []
