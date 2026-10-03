"""Multi-turn memory: SQLite checkpoint persistence for the travel planning graph.

The compiled graph saves a checkpoint of the whole TravelPlannerState after every step,
per conversation thread, in a SQLite database. A later run on the same thread reloads
the conversation (`messages`), and the orchestrator uses it to resolve follow-up queries.

Nothing is ever deleted: each turn adds new checkpoints, so the full history of every
turn - routing tasks, agents' tool calls and tool results, final answers - remains
available for investigation and audit (see get_audit_trail).

To move to Postgres later, replace AsyncSqliteSaver in open_checkpointer() with
AsyncPostgresSaver from langgraph-checkpoint-postgres; the rest of this module only
uses the generic LangGraph checkpointer API.
"""

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

import aiosqlite
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# Override with the TRAVEL_PLANNER_DB_PATH env var or the CLI's --db option
DEFAULT_DB_PATH = "data/travel_planner_memory.sqlite"
DEFAULT_THREAD_ID = "default"

# Characters of each message/result shown in an audit trail summary
AUDIT_PREVIEW_CHARS = 120

# Our own types stored in checkpoints. Listing them puts the serializer in strict mode:
# only LangGraph's built-in safe types plus these are deserialized from the database.
CHECKPOINT_TYPES = [("st.core.state", "AgentTask")]


def resolve_db_path(db_path: Optional[str] = None) -> Path:
    """Database location: explicit path, else $TRAVEL_PLANNER_DB_PATH, else the default."""
    return Path(db_path or os.getenv("TRAVEL_PLANNER_DB_PATH") or DEFAULT_DB_PATH)


@asynccontextmanager
async def open_checkpointer(db_path: Optional[str] = None) -> AsyncIterator[AsyncSqliteSaver]:
    """Open the SQLite checkpointer (creating the database file if needed).

    Args:
        db_path: Optional database path (see resolve_db_path)

    Yields:
        AsyncSqliteSaver to pass to build_parallel_travel_agent(checkpointer=...)
    """
    path = resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
    async with aiosqlite.connect(str(path)) as conn:
        yield AsyncSqliteSaver(conn, serde=serde)


def thread_config(thread_id: str = DEFAULT_THREAD_ID) -> dict:
    """LangGraph run config selecting a conversation thread."""
    return {"configurable": {"thread_id": thread_id}}


def new_turn_input(query: str, new_conversation: bool = False) -> dict:
    """Graph input for one conversation turn.

    The orchestrator clears the previous turn's working state (tasks, agent results,
    agent message histories); the conversation in `messages` is kept unless
    `new_conversation` is set. Older checkpoints are never modified either way.

    Args:
        query: The user's query for this turn
        new_conversation: Start the thread's conversation over

    Returns:
        Input dict for graph.ainvoke
    """
    messages: List[AnyMessage] = [RemoveMessage(id=REMOVE_ALL_MESSAGES)] if new_conversation else []
    return {"user_query": query, "messages": messages + [HumanMessage(content=query)]}


async def get_conversation(graph, thread_id: str = DEFAULT_THREAD_ID) -> List[AnyMessage]:
    """Current conversation (user queries and final answers) of a thread."""
    snapshot = await graph.aget_state(thread_config(thread_id))
    return snapshot.values.get("messages", [])


@dataclass
class AuditEntry:
    """One node execution recorded in a thread's checkpoint history."""
    step: int
    node: str
    created_at: str
    checkpoint_id: str
    details: List[str]


def _preview(text: str, limit: Optional[int]) -> str:
    text = " ".join(str(text).split())
    if limit is not None and len(text) > limit:
        return text[:limit] + " ..."
    return text


def _describe_message(message: Any, limit: Optional[int]) -> str:
    if isinstance(message, RemoveMessage):
        return "cleared"
    if not isinstance(message, BaseMessage):
        return _preview(message, limit)
    if getattr(message, "tool_calls", None):
        calls = ", ".join(
            f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in message.tool_calls
        )
        return f"{message.type} tool calls: {calls}"
    return f"{message.type} ({len(message.text)} chars): {_preview(message.text, limit)}"


def describe_writes(writes: dict, limit: Optional[int] = AUDIT_PREVIEW_CHARS) -> List[str]:
    """Human-readable lines describing what one node wrote to the state."""
    lines = []
    for key, value in writes.items():
        if key in ("messages", "search_messages", "planner_messages"):
            lines += [f"{key}: {_describe_message(m, limit)}" for m in value]
        elif key == "tasks":
            lines += [f"task {t.source}: {_preview(t.user_query, limit)}" for t in value]
        elif key == "agent_results":
            if value is None:
                lines.append("agent_results: cleared")
            else:
                lines += [
                    f"agent_result {r['agent']} ({len(r['result'])} chars): {_preview(r['result'], limit)}"
                    for r in value
                ]
        elif key == "final_answer":
            if value:
                lines.append(f"final_answer ({len(value)} chars): {_preview(value, limit)}")
        elif key == "user_query":
            lines.append(f"user_query: {_preview(value, limit)}")
        else:
            lines.append(f"{key}: {_preview(value, limit)}")
    return lines


async def get_audit_trail(
    graph,
    thread_id: str = DEFAULT_THREAD_ID,
    preview_chars: Optional[int] = AUDIT_PREVIEW_CHARS,
) -> List[AuditEntry]:
    """Every node execution stored for a thread, oldest first, across all turns.

    Args:
        graph: Graph compiled with a checkpointer
        thread_id: Conversation thread
        preview_chars: Truncate message/result text to this many characters (None = full text)

    Returns:
        List of AuditEntry
    """
    history = [snapshot async for snapshot in graph.aget_state_history(thread_config(thread_id))]
    entries = []
    for snapshot in reversed(history):
        for task in snapshot.tasks:
            if task.result is None:
                continue
            entries.append(AuditEntry(
                step=snapshot.metadata.get("step"),
                node="input" if task.name == "__start__" else task.name,
                created_at=snapshot.created_at,
                checkpoint_id=snapshot.config["configurable"]["checkpoint_id"],
                details=describe_writes(task.result, preview_chars),
            ))
    return entries
