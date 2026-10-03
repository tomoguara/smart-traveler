"""Synthesizer for combining results from parallel agents."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from ..core.state import TravelPlannerState


def create_synthesizer(llm: ChatOpenAI):
    """Create a synthesizer node for combining agent results.

    Args:
        llm: Chat model instance

    Returns:
        synthesizer_node function
    """
    def synthesizer_node(state: TravelPlannerState):
        """Synthesizes results from multiple agents into a unified response.

        This node:
        1. Collects results from all parallel agents
        2. Combines them into a coherent response
        3. Highlights key information and recommendations
        4. Addresses any conflicts or alternatives

        Args:
            state: Current state with agent_results from parallel agents

        Returns:
            Updates to state with final_answer and synthesized messages
        """
        agent_results = state.get("agent_results", [])

        # If no results yet, return empty
        if not agent_results:
            return {
                "final_answer": "I'm still gathering information. Please try again.",
                "messages": [AIMessage(content="Gathering information...")]
            }

        # Format results for synthesis
        results_formatted = "\n\n" + "="*50 + "\n\n".join([
            f"**{r['agent'].upper()}**\nFocus: {r.get('focus', 'N/A')}\n\n{r['result']}"
            for r in agent_results
        ])

        synthesis_prompt = """You are a Travel Response Synthesizer. Combine multiple agent outputs into
a single, well-organized, comprehensive response.

RULES:
1. Organize logically (e.g., Flights → Hotels → Itinerary → Tips), but ONLY include sections
   the agent results actually cover - never add empty or placeholder sections
2. Never invent flights, hotels or prices that are not in the agent results; don't repeat information
3. Highlight key recommendations
4. Note any conflicts or alternatives
5. Create clear sections with headers
6. End with actionable next steps

Original Query: {query}

Agent Results:
{results}

Create a unified, helpful response:"""

        response = llm.invoke([
            SystemMessage(content=synthesis_prompt.format(
                query=state["user_query"],
                results=results_formatted
            ))
        ])

        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content)]
        }

    return synthesizer_node
