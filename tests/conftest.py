"""Shared test doubles for the travel planning multi-agent system."""

import importlib
import pytest
from langchain_core.messages import AIMessage

# importlib: st.tools re-exports the deep_research tool, shadowing the submodule name
deep_research_module = importlib.import_module("st.tools.deep_research")


class StubResearchAgent:
    """Stands in for the compiled deepagents sub-agent."""

    def __init__(self, reply="Day 1: research result", error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append((payload, config))
        if self.error:
            raise self.error
        return {"messages": [AIMessage(content="intermediate step"), AIMessage(content=self.reply)]}


class StubLLM:
    """Chat model double: records bound tools and replays scripted responses."""

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.bound_tools = None
        self.seen = []

    def bind_tools(self, tools):
        self.bound_tools = [t.name for t in tools]
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def clear_research_agent_cache():
    """Each test builds its own research sub-agent."""
    deep_research_module.get_itinerary_research_agent.cache_clear()
    yield
    deep_research_module.get_itinerary_research_agent.cache_clear()


@pytest.fixture
def stub_research_agent(monkeypatch):
    """Replace the deepagents sub-agent with a StubResearchAgent (no network)."""

    def install(**kwargs):
        agent = StubResearchAgent(**kwargs)
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        monkeypatch.setattr(deep_research_module, "get_itinerary_research_agent", lambda: agent)
        return agent

    return install
