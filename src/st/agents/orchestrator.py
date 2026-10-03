"""Orchestrator agent for routing queries to appropriate sub-agents."""

from datetime import date
from typing import Dict, Any, List, Optional
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from ..core.state import TravelPlannerState, AgentTask, ClassificationResult

# How much earlier conversation the router sees when resolving follow-up queries
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_MESSAGE_CHARS = 1500

classification_prompt = """You are a Travel Request Router for a multi-agent travel planning system.
Today's date is {today}.

AVAILABLE AGENTS:
1. search_agent - flight and hotel searches: options, prices, availability
2. planner_agent - itineraries, activities, travel tips, weather, visas, general travel questions

ROUTING RULES:
- Flight and/or hotel requests ONLY -> one task for search_agent
- Itinerary, planning or general travel questions ONLY -> one task for planner_agent
- Requests that need BOTH (e.g. "find flights and plan my trip") -> one task for EACH agent
- Never create more than one task per agent

WRITING EACH TASK'S user_query (CRITICAL):
- Each agent sees ONLY its own user_query, so make it a complete, self-contained request.
- search_agent: ONLY the flight/hotel parts - origin, destination, dates, travellers, rooms,
  budget and preferences. Nothing about itineraries or activities.
- planner_agent: ONLY the planning parts - destination, dates or trip length, traveller
  interests, pace and budget. Never mention flight or hotel searches.
- Convert relative dates ("next Friday", "in two weeks") into absolute YYYY-MM-DD dates using
  today's date. Do not invent dates the user did not give.
- focus: a few words describing what the agent is responsible for in this request.

EXAMPLES:
Query: "Find flights from DEL to DXB on 2026-12-03"
-> tasks: [{{"source": "search_agent", "user_query": "Find flights from DEL to DXB on 2026-12-03", "focus": "flight options and prices"}}]

Query: "Plan a 5-day trip to Paris"
-> tasks: [{{"source": "planner_agent", "user_query": "Plan a 5-day itinerary for Paris", "focus": "daily itinerary"}}]

Query: "Plan a week in London from 2026-11-02 and find me hotels under $200/night"
-> tasks: [
  {{"source": "search_agent", "user_query": "Find hotels in London under $200/night, check-in 2026-11-02, check-out 2026-11-09", "focus": "hotel options and prices"}},
  {{"source": "planner_agent", "user_query": "Plan a 7-day itinerary for London from 2026-11-02 to 2026-11-09", "focus": "daily activities and attractions"}}
]"""

conversation_prompt = """

CONVERSATION SO FAR (oldest first):
{history}

The new query may be a follow-up to this conversation. Use it to resolve references such as
"there", "those dates" or "the same trip", and make every sub-query fully self-contained
(repeat the destination, dates, travellers and preferences from earlier turns when they apply).
If the new query is only about the conversation itself (e.g. "what did I ask before?"), create
one planner_agent task whose user_query quotes the relevant earlier details."""


def format_history(messages: List[AnyMessage]) -> str:
    """Render recent conversation turns for the router prompt."""
    lines = []
    for message in messages[-MAX_HISTORY_MESSAGES:]:
        role = "User" if message.type == "human" else "Assistant"
        text = message.text
        if len(text) > MAX_HISTORY_MESSAGE_CHARS:
            text = text[:MAX_HISTORY_MESSAGE_CHARS] + " ..."
        lines.append(f"{role}: {text}")
    return "\n\n".join(lines)


def _merge_tasks(tasks: List[AgentTask]) -> List[AgentTask]:
    """Keep one task per agent, merging duplicates so no part of the request is lost."""
    merged: Dict[str, AgentTask] = {}
    for task in tasks:
        if task.source in merged:
            first = merged[task.source]
            merged[task.source] = AgentTask(
                source=task.source,
                user_query=f"{first.user_query}\n{task.user_query}",
                focus=first.focus or task.focus,
            )
        else:
            merged[task.source] = task
    return list(merged.values())


class QueryClassifier:
    """Classifies user queries and writes a focused sub-query for each agent."""

    def __init__(self, llm: ChatOpenAI):
        """Initialize the query classifier.

        Args:
            llm: Chat model instance for classification
        """
        self.llm = llm

    def classify_query(
        self,
        query: str,
        today: Optional[date] = None,
        history: Optional[List[AnyMessage]] = None,
    ) -> ClassificationResult:
        """Classify a user query to determine which agents should handle it.

        Args:
            query: User's input query
            today: Date used to resolve relative dates (defaults to today)
            history: Earlier conversation messages, used to resolve follow-up queries

        Returns:
            ClassificationResult with one task per agent to invoke
        """
        system_prompt = classification_prompt.format(today=(today or date.today()).isoformat())
        if history:
            system_prompt += conversation_prompt.format(history=format_history(history))

        structured_llm = self.llm.with_structured_output(ClassificationResult)
        result = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ])

        tasks = _merge_tasks(result.tasks)
        if not tasks:
            # Nothing travel-specific identified: let the planner handle it
            tasks = [AgentTask(source="planner_agent", user_query=query, focus="general travel information")]

        return ClassificationResult(tasks=tasks, requires_synthesis=len(tasks) > 1)


def classify_query_parallel(state: TravelPlannerState) -> Dict[str, Any]:
    """Orchestrator that classifies queries and dispatches to appropriate agents.

    This node starts every turn:
    1. Receives the user query (plus earlier conversation, when the thread has memory)
    2. Classifies the query type with structured output
    3. Creates a focused task (sub-query) for each agent to invoke
    4. Clears the previous turn's per-turn channels and returns updates for routing

    Args:
        state: Current state with user_query and the conversation so far

    Returns:
        State updates dict with tasks, requires_synthesis and per-turn resets
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    classifier = QueryClassifier(llm)

    # Earlier conversation = all messages before this turn's query
    messages = state.get("messages", [])
    history = messages[:-1] if messages and messages[-1].type == "human" else messages

    try:
        classification_result = classifier.classify_query(state["user_query"], history=history)
    except Exception:
        # Classification failed: fall back to both agents with the full query
        classification_result = ClassificationResult(
            tasks=[
                AgentTask(source="search_agent", user_query=state["user_query"], focus="flights and hotels"),
                AgentTask(source="planner_agent", user_query=state["user_query"], focus="travel planning and itinerary"),
            ],
            requires_synthesis=True,
        )

    return {
        "tasks": classification_result.tasks,
        "requires_synthesis": classification_result.requires_synthesis,
        # New turn: clear the previous turn's working state (older checkpoints keep it for audit)
        "agent_results": None,
        "search_messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        "planner_messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        "final_answer": "",
    }
