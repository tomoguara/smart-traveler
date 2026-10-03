"""Live test for the deep_research tool (real OpenAI + Tavily calls).

Skipped by default; run with: uv run pytest -m live
"""

import os

import pytest

from st.tools import deep_research

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") and os.getenv("TAVILY_API_KEY")),
    reason="OPENAI_API_KEY and TAVILY_API_KEY are required",
)
def test_deep_research_returns_day_by_day_itinerary():
    result = deep_research.invoke({
        "query": "Plan a 2-day itinerary in Lisbon for a food and history lover on a mid-range budget"
    })

    assert not result.startswith("Error"), result
    assert "day 1" in result.lower()
    assert "day 2" in result.lower()
    assert len(result) > 500
