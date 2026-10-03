"""State definitions for the Multi-Agent Travel Planning System.

This module defines the shared state that all agents read from and write to.
The state carries the evolving context as the graph executes. With a checkpointer,
the state of a conversation thread is persisted between runs (multi-turn memory):
`messages` keeps the conversation, while the per-turn channels are cleared by the
orchestrator at the start of every turn (older checkpoints keep them for auditing).
"""

from typing import TypedDict, Annotated, List, Literal, Optional
from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


def append_or_reset(left: List[dict], right: Optional[List[dict]]) -> List[dict]:
    """Reducer: append new items (concurrent writes are merged); `None` clears the list."""
    if right is None:
        return []
    return left + right


class AgentTask(BaseModel):
    """A single agent task with targeted query."""
    source: Literal["search_agent", "planner_agent"] = Field(
        description="Agent that handles this task"
    )
    user_query: str = Field(
        description="Self-contained sub-query containing only this agent's part of the request"
    )
    focus: str = Field(
        default="",
        description="Short description of what this agent should focus on"
    )


class TravelPlannerState(TypedDict):
    """Shared state for the multi-agent travel planner.

    This TypedDict defines the contract that all agents must follow.
    Each agent reads from and writes to this shared state.
    """

    # Conversation history across turns (user queries + final answers)
    messages: Annotated[List[AnyMessage], add_messages]

    # Current user query
    user_query: str

    # Tasks from router (supports parallel execution)
    tasks: List[AgentTask]

    # Flag indicating if synthesis is needed (multiple agents)
    requires_synthesis: bool

    # Parallel agent results for the current turn - appends concurrent writes; None clears
    agent_results: Annotated[List[dict], append_or_reset]

    # Final synthesized answer
    final_answer: str

    # Agent-specific message histories for the current turn
    # (add_messages: RemoveMessage(id=REMOVE_ALL_MESSAGES) clears them)
    search_messages: Annotated[List[AnyMessage], add_messages]
    planner_messages: Annotated[List[AnyMessage], add_messages]


def get_task(state: TravelPlannerState, source: str) -> Optional[AgentTask]:
    """Return the orchestrator's task for an agent, if one was created."""
    for task in state.get("tasks", []):
        if task.source == source:
            return task
    return None


class ClassificationResult(BaseModel):
    """Router output - supports MULTIPLE parallel tasks."""
    tasks: List[AgentTask] = Field(
        description="List of agents to invoke with their targeted queries"
    )
    requires_synthesis: bool = Field(
        default=False,
        description="Whether multiple agents are being used and synthesis is needed"
    )
