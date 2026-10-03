# Copyright (C) 2026 Tonworio(Ton) Oguara
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the deep_research tool (no network calls)."""

import importlib
import deepagents
import pytest
from langchain_core.messages import AIMessage

from st.agents.prompts import research_instructions
from st.tools import deep_research

# importlib: st.tools re-exports the deep_research tool, shadowing the submodule name
deep_research_module = importlib.import_module("st.tools.deep_research")


def test_tool_schema():
    assert deep_research.name == "deep_research"
    assert set(deep_research.args) == {"query"}


def test_returns_final_sub_agent_message(stub_research_agent):
    agent = stub_research_agent(reply="Day 1: Alfama and Baixa")

    result = deep_research.invoke({"query": "3-day Lisbon itinerary for food lovers"})

    assert result == "Day 1: Alfama and Baixa"
    payload, config = agent.calls[0]
    assert payload == {"messages": [{"role": "user", "content": "3-day Lisbon itinerary for food lovers"}]}
    assert config == {"recursion_limit": deep_research_module.DEEP_RESEARCH_RECURSION_LIMIT}


def test_content_blocks_are_flattened_to_text(stub_research_agent):
    agent = stub_research_agent()
    agent.invoke = lambda payload, config=None: {
        "messages": [AIMessage(content=[{"type": "text", "text": "Day 1: Belem"}])]
    }

    assert deep_research.invoke({"query": "Lisbon"}) == "Day 1: Belem"


def test_sub_agent_failure_is_returned_as_text(stub_research_agent):
    stub_research_agent(error=RuntimeError("rate limited"))

    result = deep_research.invoke({"query": "Lisbon"})

    assert result == "Error during deep research: rate limited"


def test_missing_tavily_key_raises(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        deep_research.invoke({"query": "Lisbon"})


def _capture_create_deep_agent(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(deepagents, "create_deep_agent", fake_create_deep_agent)
    return captured


def test_sub_agent_wiring_uses_default_model(monkeypatch):
    captured = _capture_create_deep_agent(monkeypatch)
    monkeypatch.delenv("DEEP_RESEARCH_MODEL", raising=False)

    agent = deep_research_module.get_itinerary_research_agent()

    assert captured["model"].model_name == "gpt-4o-mini"
    assert [t.name for t in captured["tools"]] == ["tavily_search"]
    assert captured["tools"][0].search_depth == "advanced"
    assert captured["system_prompt"] == research_instructions
    # Built once and reused
    assert deep_research_module.get_itinerary_research_agent() is agent


def test_sub_agent_model_is_configurable(monkeypatch):
    captured = _capture_create_deep_agent(monkeypatch)
    monkeypatch.setenv("DEEP_RESEARCH_MODEL", "gpt-5.2")

    deep_research_module.get_itinerary_research_agent()

    assert captured["model"].model_name == "gpt-5.2"
