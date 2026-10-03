"""Tests for multi-turn memory: SQLite checkpoint persistence, per-turn resets and the CLI (no network)."""

import asyncio
import builtins
import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

import st.agents.orchestrator as orchestrator_module
import st.main as main_module
from st.core import AgentTask, ClassificationResult
from st.core.state import append_or_reset
from st.memory import get_audit_trail, new_turn_input, open_checkpointer, resolve_db_path


class RouterStub:
    """Orchestrator model double: one planner task per turn, records the router prompts."""

    def __init__(self):
        self.system_prompts = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        system, human = messages
        self.system_prompts.append(system.content)
        return ClassificationResult(tasks=[AgentTask(source="planner_agent", user_query=human.content)])


class AgentStub:
    """Planner + synthesizer model double: answers without tool calls."""

    def __init__(self):
        self.answers = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        system = messages[0].content if isinstance(messages[0], SystemMessage) else ""
        if "travel scout" in system:
            return AIMessage(content=f"PLAN for: {messages[-1].content}")
        self.answers += 1
        return AIMessage(content=f"ANSWER {self.answers}")


@pytest.fixture
def stubs(monkeypatch, tmp_path):
    """Stub every model the CLI builds; returns (router, agent_llm, db_path)."""
    router, agent_llm = RouterStub(), AgentStub()
    monkeypatch.setattr(orchestrator_module, "ChatOpenAI", lambda **kwargs: router)
    monkeypatch.setattr(main_module, "ChatOpenAI", lambda **kwargs: agent_llm)
    return router, agent_llm, str(tmp_path / "memory.sqlite")


def turn(query, db, thread_id="trip", new_conversation=False):
    return asyncio.run(main_module.run_travel_planning_workflow(query, thread_id, new_conversation, db))


def audit(db, thread_id="trip"):
    async def collect():
        async with open_checkpointer(db) as checkpointer:
            return await get_audit_trail(main_module.create_travel_planner(checkpointer), thread_id)
    return asyncio.run(collect())


# --- building blocks ----------------------------------------------------------

def test_append_or_reset_reducer():
    assert append_or_reset([{"a": 1}], [{"b": 2}]) == [{"a": 1}, {"b": 2}]
    assert append_or_reset([{"a": 1}], None) == []


def test_new_turn_input():
    plain = new_turn_input("Plan Lisbon")
    assert plain["user_query"] == "Plan Lisbon"
    assert [(m.type, m.content) for m in plain["messages"]] == [("human", "Plan Lisbon")]

    fresh = new_turn_input("Plan Lisbon", new_conversation=True)
    assert isinstance(fresh["messages"][0], RemoveMessage)
    assert fresh["messages"][0].id == REMOVE_ALL_MESSAGES


def test_db_path_resolution(monkeypatch):
    monkeypatch.delenv("TRAVEL_PLANNER_DB_PATH", raising=False)
    assert str(resolve_db_path()) == "data/travel_planner_memory.sqlite"
    monkeypatch.setenv("TRAVEL_PLANNER_DB_PATH", "/tmp/x.sqlite")
    assert str(resolve_db_path()) == "/tmp/x.sqlite"
    assert str(resolve_db_path("other.sqlite")) == "other.sqlite"


# --- multi-turn behaviour -----------------------------------------------------

def test_follow_up_turn_sees_history_and_starts_clean(stubs):
    router, _, db = stubs

    turn("Plan 3 days in Lisbon", db)
    result = turn("Now add a day trip there", db)  # new process-equivalent: fresh checkpointer

    # The router saw the earlier turn and can resolve "there"
    assert "CONVERSATION SO FAR" not in router.system_prompts[0]
    assert "User: Plan 3 days in Lisbon" in router.system_prompts[1]
    assert "Assistant: ANSWER 1" in router.system_prompts[1]

    # Conversation persists; per-turn working state holds only this turn
    assert [(m.type, m.content) for m in result["messages"]] == [
        ("human", "Plan 3 days in Lisbon"), ("ai", "ANSWER 1"),
        ("human", "Now add a day trip there"), ("ai", "ANSWER 2"),
    ]
    assert [r["result"] for r in result["agent_results"]] == ["PLAN for: Now add a day trip there"]
    assert [m.content for m in result["planner_messages"]] == [
        "Now add a day trip there", "PLAN for: Now add a day trip there"]
    assert result["final_answer"] == "ANSWER 2"


def test_threads_are_isolated(stubs):
    router, _, db = stubs

    turn("Plan 3 days in Lisbon", db, thread_id="lisbon")
    result = turn("Plan 2 days in Rome", db, thread_id="rome")

    assert "CONVERSATION SO FAR" not in router.system_prompts[1]
    assert [m.content for m in result["messages"]] == ["Plan 2 days in Rome", "ANSWER 2"]


def test_new_conversation_keeps_earlier_turns_for_audit(stubs):
    router, _, db = stubs

    turn("Plan 3 days in Lisbon", db)
    result = turn("Plan 2 days in Rome", db, new_conversation=True)

    assert "CONVERSATION SO FAR" not in router.system_prompts[1]
    assert [m.content for m in result["messages"]] == ["Plan 2 days in Rome", "ANSWER 2"]

    # Nothing was deleted: the first turn is still in the checkpoint history
    lines = [line for entry in audit(db) for line in entry.details]
    assert "user_query: Plan 3 days in Lisbon" in lines
    assert any(line.startswith("agent_result planner_agent") and "Lisbon" in line for line in lines)
    assert any(line.startswith("final_answer") and "ANSWER 1" in line for line in lines)


def test_audit_trail_lists_every_node_in_order(stubs):
    _, _, db = stubs

    turn("Plan 3 days in Lisbon", db)
    entries = audit(db)

    assert [e.node for e in entries] == ["input", "orchestrator", "planner_agent", "synthesizer"]
    assert [e.step for e in entries] == sorted(e.step for e in entries)
    orchestrator_lines = entries[1].details
    assert "task planner_agent: Plan 3 days in Lisbon" in orchestrator_lines
    assert "agent_results: cleared" in orchestrator_lines


# --- CLI ----------------------------------------------------------------------

def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["st", *args])
    main_module.main()


def test_cli_turns_history_and_audit(stubs, monkeypatch, capsys):
    _, _, db = stubs

    run_cli(monkeypatch, "Plan 3 days in Lisbon", "--thread-id", "trip", "--db", db)
    run_cli(monkeypatch, "Now add a day trip there", "--thread-id", "trip", "--db", db)
    capsys.readouterr()

    run_cli(monkeypatch, "--history", "--thread-id", "trip", "--db", db)
    history = capsys.readouterr().out
    assert history.index("Plan 3 days in Lisbon") < history.index("ANSWER 1") \
        < history.index("Now add a day trip there") < history.index("ANSWER 2")

    run_cli(monkeypatch, "--audit", "--thread-id", "trip", "--db", db)
    audit_out = capsys.readouterr().out
    assert audit_out.count("] orchestrator") == 2
    assert audit_out.count("] synthesizer") == 2


def test_cli_chat_saves_every_turn(stubs, monkeypatch, capsys):
    _, _, db = stubs
    replies = iter(["Plan 3 days in Lisbon", "", "Now add a day trip there", "exit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(replies))

    run_cli(monkeypatch, "--chat", "--thread-id", "chat", "--db", db)
    run_cli(monkeypatch, "--history", "--thread-id", "chat", "--db", db)

    out = capsys.readouterr().out
    assert "ANSWER 1" in out and "ANSWER 2" in out
    assert "[You]\nNow add a day trip there" in out


def test_cli_history_of_unknown_thread(stubs, monkeypatch, capsys):
    _, _, db = stubs
    run_cli(monkeypatch, "--history", "--thread-id", "nobody", "--db", db)
    assert "No conversation saved for thread 'nobody'" in capsys.readouterr().out


def test_cli_rejects_query_with_history(monkeypatch):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "Plan Lisbon", "--history")
