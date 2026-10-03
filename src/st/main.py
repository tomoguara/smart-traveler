# Copyright (C) 2026 Tonworio(Ton) Oguara
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Main entry point for the Multi-Agent Travel Planning System.

This module provides a CLI interface to run the travel planning workflow
with user queries, demonstrating the multi-agent system end-to-end.

Conversations are persisted per thread in a SQLite database (multi-turn memory),
so a later run on the same thread can ask follow-up questions.
"""

import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI

from st.graph.builder import build_parallel_travel_agent
from st.memory import (
    DEFAULT_THREAD_ID,
    get_audit_trail,
    get_conversation,
    new_turn_input,
    open_checkpointer,
    resolve_db_path,
    thread_config,
)

DEFAULT_QUERY = "Plan a 5-day trip to Paris with flights and hotels"


def create_travel_planner(checkpointer=None):
    """Create a complete travel planning agent workflow.

    Args:
        checkpointer: Optional checkpointer for multi-turn memory (see st.memory)
    """

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Build graph (creates agents internally)
    graph = build_parallel_travel_agent(llm, checkpointer=checkpointer)

    return graph


async def run_turn(graph, query: str, thread_id: str = DEFAULT_THREAD_ID, new_conversation: bool = False):
    """Run one conversation turn on a graph compiled with a checkpointer.

    Args:
        graph: Compiled travel planning graph
        query: The user's travel-related query
        thread_id: Conversation thread to continue
        new_conversation: Start the thread's conversation over (history is kept for audit)

    Returns:
        The final state of the turn
    """
    print("🚀 Starting travel planning workflow...")
    print(f"🧵 Thread: {thread_id}{' (new conversation)' if new_conversation else ''}")
    print(f"📝 Query: {query}\n")

    result = await graph.ainvoke(new_turn_input(query, new_conversation), thread_config(thread_id))

    print("\n✅ Workflow completed!")
    print(f"\n🎯 Final Answer:\n{result['final_answer']}")

    return result


async def run_travel_planning_workflow(
    query: str,
    thread_id: str = DEFAULT_THREAD_ID,
    new_conversation: bool = False,
    db_path: str = None,
):
    """Run the travel planning workflow with a user query.

    Args:
        query: The user's travel-related query
        thread_id: Conversation thread to continue
        new_conversation: Start the thread's conversation over
        db_path: SQLite database for conversation memory

    Returns:
        The final state of the turn
    """
    async with open_checkpointer(db_path) as checkpointer:
        compiled_workflow = create_travel_planner(checkpointer)
        return await run_turn(compiled_workflow, query, thread_id, new_conversation)


async def chat(thread_id: str = DEFAULT_THREAD_ID, new_conversation: bool = False, db_path: str = None):
    """Interactive chat loop; every turn is saved to the same thread."""
    async with open_checkpointer(db_path) as checkpointer:
        compiled_workflow = create_travel_planner(checkpointer)
        print(f"💬 Travel planner chat on thread '{thread_id}' (type 'exit' or press Ctrl-D to quit)")

        while True:
            try:
                query = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if query.lower() in ("exit", "quit"):
                break
            if not query:
                continue

            await run_turn(compiled_workflow, query, thread_id, new_conversation)
            new_conversation = False  # only the first turn starts over


async def show_history(thread_id: str = DEFAULT_THREAD_ID, db_path: str = None):
    """Print the saved conversation of a thread."""
    async with open_checkpointer(db_path) as checkpointer:
        messages = await get_conversation(create_travel_planner(checkpointer), thread_id)

    if not messages:
        print(f"No conversation saved for thread '{thread_id}'.")
        return
    print(f"🧵 Conversation on thread '{thread_id}':")
    for message in messages:
        role = "You" if message.type == "human" else "Planner"
        print(f"\n[{role}]\n{message.text}")


async def show_audit(thread_id: str = DEFAULT_THREAD_ID, db_path: str = None, full: bool = False):
    """Print every stored step of a thread (all turns), oldest first."""
    async with open_checkpointer(db_path) as checkpointer:
        entries = await get_audit_trail(
            create_travel_planner(checkpointer), thread_id, preview_chars=None if full else 120
        )

    if not entries:
        print(f"No checkpoints saved for thread '{thread_id}'.")
        return
    print(f"🔎 Audit trail for thread '{thread_id}' ({len(entries)} steps, {resolve_db_path(db_path)}):")
    for entry in entries:
        print(f"\n[step {entry.step}] {entry.node}  @ {entry.created_at}  (checkpoint {entry.checkpoint_id})")
        for line in entry.details:
            print(f"    {line}")


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Agent Travel Planning System",
        epilog=(
            'Examples: st "Find me flights from NYC to London on 2026-11-10"  |  '
            'st --thread-id lisbon "Now find hotels there"  |  st --chat  |  st --history'
        ),
    )
    parser.add_argument(
        "query",
        nargs="?",
        help=f'Your travel query (default: "{DEFAULT_QUERY}")'
    )
    parser.add_argument(
        "--thread-id",
        default=DEFAULT_THREAD_ID,
        help="Conversation thread to continue or create (default: %(default)s)"
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Start the thread's conversation over (earlier turns stay stored for --audit)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--chat", action="store_true", help="Interactive multi-turn chat")
    mode.add_argument("--history", action="store_true", help="Print the thread's saved conversation")
    mode.add_argument("--audit", action="store_true", help="Print every stored step of the thread (all turns)")
    parser.add_argument("--full", action="store_true", help="With --audit: show full text instead of previews")
    parser.add_argument(
        "--db",
        help="SQLite database for conversation memory "
             "(default: $TRAVEL_PLANNER_DB_PATH or data/travel_planner_memory.sqlite)"
    )

    args = parser.parse_args()

    if args.query and (args.chat or args.history or args.audit):
        parser.error("a query cannot be combined with --chat, --history or --audit")

    # Run async workflow
    if args.chat:
        asyncio.run(chat(args.thread_id, args.new, args.db))
    elif args.history:
        asyncio.run(show_history(args.thread_id, args.db))
    elif args.audit:
        asyncio.run(show_audit(args.thread_id, args.db, args.full))
    else:
        asyncio.run(run_travel_planning_workflow(args.query or DEFAULT_QUERY, args.thread_id, args.new, args.db))


if __name__ == "__main__":
    main()
