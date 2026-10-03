"""Unit tests for orchestrator routing, per-agent sub-queries and the deferred synthesizer (no network)."""

from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

import st.agents.orchestrator as orchestrator_module
from st.agents import QueryClassifier, classify_query_parallel, create_search_agent
from st.core import AgentTask, ClassificationResult
from st.graph import build_parallel_travel_agent

from conftest import StubLLM

MIXED_QUERY = "Find flights JFK to LIS on 2026-11-10 and plan 4 days in Lisbon for a food lover"
SEARCH_TASK = AgentTask(source="search_agent", user_query="Find flights from JFK to LIS on 2026-11-10",
                        focus="flight options")
PLANNER_TASK = AgentTask(source="planner_agent", user_query="Plan a 4-day food itinerary in Lisbon",
                         focus="food itinerary")


class StructuredStub:
    """Chat model double for with_structured_output: returns a fixed result or raises."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.schema = None
        self.seen = []

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        if self.error:
            raise self.error
        return self.result


def _patch_orchestrator(monkeypatch, stub):
    monkeypatch.setattr(orchestrator_module, "ChatOpenAI", lambda **kwargs: stub)


# --- orchestrator -----------------------------------------------------------

def test_classifier_uses_structured_output_with_todays_date():
    stub = StructuredStub(ClassificationResult(tasks=[SEARCH_TASK, PLANNER_TASK]))

    result = QueryClassifier(stub).classify_query(MIXED_QUERY, today=date(2026, 10, 2))

    assert stub.schema is ClassificationResult
    system, human = stub.seen[0]
    assert "Today's date is 2026-10-02" in system.content
    assert human == HumanMessage(content=MIXED_QUERY)
    assert result.tasks == [SEARCH_TASK, PLANNER_TASK]
    assert result.requires_synthesis is True


def test_classifier_merges_duplicate_tasks_per_agent():
    extra = AgentTask(source="search_agent", user_query="Find hotels in Lisbon 2026-11-10 to 2026-11-14")
    stub = StructuredStub(ClassificationResult(tasks=[SEARCH_TASK, extra]))

    result = QueryClassifier(stub).classify_query(MIXED_QUERY)

    [task] = result.tasks
    assert task.user_query == f"{SEARCH_TASK.user_query}\n{extra.user_query}"
    assert task.focus == "flight options"
    assert result.requires_synthesis is False


def test_classifier_without_tasks_falls_back_to_planner():
    stub = StructuredStub(ClassificationResult(tasks=[]))

    result = QueryClassifier(stub).classify_query("hello")

    assert result.tasks == [AgentTask(source="planner_agent", user_query="hello", focus="general travel information")]


def test_orchestrator_node_writes_tasks(monkeypatch):
    _patch_orchestrator(monkeypatch, StructuredStub(ClassificationResult(tasks=[PLANNER_TASK])))

    update = classify_query_parallel({"user_query": "Plan 4 days in Lisbon"})

    assert update["tasks"] == [PLANNER_TASK]
    assert update["requires_synthesis"] is False
    # Every turn starts by clearing the previous turn's working state
    assert update["agent_results"] is None
    assert update["final_answer"] == ""
    assert [m.id for m in update["search_messages"]] == [REMOVE_ALL_MESSAGES]
    assert [m.id for m in update["planner_messages"]] == [REMOVE_ALL_MESSAGES]


def test_orchestrator_node_falls_back_to_both_agents_on_error(monkeypatch):
    _patch_orchestrator(monkeypatch, StructuredStub(error=RuntimeError("API down")))

    update = classify_query_parallel({"user_query": MIXED_QUERY})

    assert [t.source for t in update["tasks"]] == ["search_agent", "planner_agent"]
    assert all(t.user_query == MIXED_QUERY for t in update["tasks"])
    assert update["requires_synthesis"] is True


# --- agents work on their own sub-query ------------------------------------

def test_search_agent_uses_its_task_and_keeps_query_in_history():
    llm = StubLLM([AIMessage(content="", tool_calls=[{"name": "search_flights", "args": {}, "id": "c1"}])])
    search_agent_node, _ = create_search_agent(llm)

    update = search_agent_node({"user_query": MIXED_QUERY, "tasks": [SEARCH_TASK, PLANNER_TASK],
                                "search_messages": []})

    assert llm.seen[0][1] == HumanMessage(content=SEARCH_TASK.user_query)
    assert update["search_messages"][0] == HumanMessage(content=SEARCH_TASK.user_query)
    assert update["agent_results"] == []


def test_search_agent_falls_back_to_full_query_without_task():
    llm = StubLLM([AIMessage(content="Options...")])
    search_agent_node, _ = create_search_agent(llm)

    update = search_agent_node({"user_query": MIXED_QUERY, "tasks": [], "search_messages": []})

    assert llm.seen[0][1] == HumanMessage(content=MIXED_QUERY)
    assert update["agent_results"] == [{"agent": "search_agent", "focus": "flights and hotels",
                                        "result": "Options..."}]


# --- graph: fan-out, tool loops and a single synthesis ----------------------

class RoleStubLLM:
    """Answers by role (system prompt), so parallel lanes can share it safely."""

    def __init__(self):
        self.planner_turns = 0
        self.synthesis_inputs = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        system = messages[0].content if isinstance(messages[0], SystemMessage) else ""
        if "flight and hotel search agent" in system:
            return AIMessage(content="FLIGHT OPTIONS")  # search lane finishes immediately
        if "travel scout" in system:
            self.planner_turns += 1
            if self.planner_turns == 1:  # planner lane needs a tool round first
                return AIMessage(content="", tool_calls=[{"name": "deep_research",
                                                          "args": {"query": "Lisbon"}, "id": "r1"}])
            return AIMessage(content="ITINERARY")
        self.synthesis_inputs.append(system)
        return AIMessage(content="FINAL")


def test_synthesizer_runs_once_after_both_lanes_finish(monkeypatch, stub_research_agent):
    stub_research_agent(reply="Day 1: Alfama")
    _patch_orchestrator(monkeypatch, StructuredStub(ClassificationResult(tasks=[SEARCH_TASK, PLANNER_TASK])))
    llm = RoleStubLLM()
    graph = build_parallel_travel_agent(llm)

    steps = list(graph.stream({
        "messages": [], "user_query": MIXED_QUERY, "tasks": [], "requires_synthesis": False,
        "agent_results": [], "final_answer": "", "search_messages": [], "planner_messages": [],
    }, stream_mode="updates"))

    nodes = [node for step in steps for node in step]
    assert nodes.count("synthesizer") == 1
    assert nodes[-1] == "synthesizer"
    # The single synthesis saw both agents' results
    [synthesis_prompt] = llm.synthesis_inputs
    assert "FLIGHT OPTIONS" in synthesis_prompt and "ITINERARY" in synthesis_prompt
    assert steps[-1]["synthesizer"]["final_answer"] == "FINAL"
