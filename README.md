<!--
Copyright (C) 2026 Tonworio(Ton) Oguara
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# The Smart Travel Planning Multi-Agent System

A command-line travel assistant built with [LangGraph](https://langchain-ai.github.io/langgraph/). It coordinates several specialised AI agents to:

- search **flights** (including return flights for round trips) and **hotels** with live prices
- plan **day-by-day itineraries** using deep, multi-step web research
- answer **general travel questions** (best time to visit, visas, packing, customs, …)
- run independent parts of a request **in parallel** and merge them into **one formatted recommendation**
- **remember conversations**, so you can ask follow-up questions in later runs

```
You ──► Orchestrator ──┬─► Search agent  ──► flight / hotel search (SerpAPI)  ──┐
                       └─► Planner agent ──► web search / deep research (Tavily) ┴─► Synthesizer ──► answer
```

- [Part 1 – User guide: how to use the system](#part-1--user-guide-how-to-use-the-system)
- [Part 2 – Developer guide: setting up, updating and extending the system](#part-2--developer-guide-setting-up-updating-and-extending-the-system)

---

## Part 1 – User guide: how to use the system

### 1.1 Before you start

You need:

- **Python 3.14** and **[uv](https://docs.astral.sh/uv/)** (uv can install Python for you), with the project set up as described in [2.1 Setting up your environment](#21-setting-up-your-environment). In short: run `uv sync` in the project folder.
- Three API keys in a file named `.env` in the project folder:

  ```env
  OPENAI_API_KEY=sk-...
  SERPAPI_API_KEY=...
  TAVILY_API_KEY=tvly-...
  ```

  | Key | Used for | Get one at |
  |---|---|---|
  | `OPENAI_API_KEY` | All agents' reasoning (model `gpt-4o-mini`) | https://platform.openai.com/api-keys |
  | `SERPAPI_API_KEY` | Flight and hotel search (Google Flights / Google Hotels) | https://serpapi.com/users/sign_in |
  | `TAVILY_API_KEY` | Web search and deep research | https://app.tavily.com/ |

  > `.env` holds secrets. It is listed in `.gitignore`, so never commit it.

All commands below are run from the project folder. `uv run st …` works without activating anything. If you have activated the virtual environment (`source .venv/bin/activate`), you can type just `st …`.

### 1.2 Ask a question (one-shot)

```bash
uv run st "Plan a 4-day itinerary in Lisbon for a food and history lover"
```

The system decides which agents are needed, runs them, and prints a single answer. A typical run takes 30–60 seconds. Running `st` with no query uses a built-in example ("Plan a 5-day trip to Paris with flights and hotels").

What you can ask:

| Kind of request | Example | Agents used |
|---|---|---|
| Flights | `"One-way flight from New York to Lisbon on 2026-11-10"` | Search |
| Round trip | `"Round-trip flights JFK to LIS, 2026-11-10 returning 2026-11-14"` | Search (also shows return options) |
| Hotels | `"Hotels in Lisbon from 2026-11-10 to 2026-11-14, cheapest first"` | Search |
| Itinerary | `"3-day itinerary in Rome for a history lover on a mid-range budget"` | Planner (deep research) |
| General question | `"What's the best month to visit Japan for fewer crowds?"` | Planner (web search) |
| Everything at once | `"Flights from JFK to LIS on 2026-11-10 returning 2026-11-14, hotels for those dates, and a 4-day food itinerary"` | Search + Planner in parallel |

Tips for good results:

- **Dates:** absolute dates (`2026-11-10`) and relative ones ("next Friday", "in two weeks") both work. Relative dates are converted using today's date.
- **Places:** city names are fine. The system converts them to airport codes (for example, New York searches JFK, EWR and LGA).
- **Travellers and budget:** mention them ("for 2 adults", "under $200/night") and they are passed to the searches.
- **Prices:** flight and hotel prices are in USD. Round-trip prices are totals for the whole trip.

### 1.3 Conversations and follow-up questions (memory)

Every run is saved to a **conversation thread**. Later runs on the same thread can refer back to earlier answers:

```bash
uv run st --thread-id lisbon "Plan a 3-day itinerary in Lisbon from 2026-11-10 to 2026-11-13"
uv run st --thread-id lisbon "Now find me round-trip flights from New York for those dates and a hotel there"
```

The second run understands that "those dates" means 10–13 November and "there" means Lisbon.

- Without `--thread-id`, runs go to a thread called `default`, so consecutive runs continue the same conversation.
- Use a different `--thread-id` for each trip you plan. Threads are completely separate.
- Add `--new` to start a thread's conversation over. Earlier turns are **not deleted**; they stay available to `--audit`.

### 1.4 Interactive chat

```bash
uv run st --chat                      # chat on the "default" thread
uv run st --chat --thread-id japan    # chat on a named thread
uv run st --chat --thread-id japan --new   # start that thread over, then chat
```

Type a question at the `You:` prompt and press Enter. Every turn is saved to the thread, so you can close the chat and continue later, either with `--chat` or with one-shot commands. To leave, type `exit` or `quit`, or press Ctrl-D (or Ctrl-C).

### 1.5 Reviewing past conversations

```bash
uv run st --history --thread-id lisbon         # the conversation: your questions and the answers
uv run st --audit --thread-id lisbon           # every step of every turn: routing, tool calls, results
uv run st --audit --full --thread-id lisbon    # same, with full text instead of 120-character previews
```

`--audit` is for checking *how* an answer was produced. It shows, step by step, which agents ran, what each one searched for (for example, `search_flights({"departure_airport": "JFK,EWR,LGA", ...})`), and what came back.

### 1.6 CLI reference

```
st [query] [--thread-id ID] [--new] [--chat | --history | --audit] [--full] [--db PATH]
```

| Option | What it does |
|---|---|
| `query` | Your travel request (in quotes). Not used with `--chat`, `--history` or `--audit`. |
| `--thread-id ID` | Conversation thread to continue or create (default: `default`). |
| `--new` | Start the thread's conversation over. Earlier turns stay stored. |
| `--chat` | Interactive multi-turn chat. |
| `--history` | Print the thread's saved conversation. |
| `--audit` | Print every stored step of the thread, across all turns. |
| `--full` | With `--audit`: show full text instead of previews. |
| `--db PATH` | Use a different memory database (default: `data/travel_planner_memory.sqlite`). |
| `-h`, `--help` | Show the built-in help. |

Optional environment variables (put them in `.env` or your shell):

| Variable | Default | Purpose |
|---|---|---|
| `TRAVEL_PLANNER_DB_PATH` | `data/travel_planner_memory.sqlite` | Where conversations are stored (`--db` takes precedence). |
| `DEEP_RESEARCH_MODEL` | `gpt-4o-mini` | OpenAI model for the deep-research sub-agent (for example, `gpt-5.2` for higher quality at higher cost). |

### 1.7 Costs and limits

Each run calls paid APIs:

- **OpenAI:** several `gpt-4o-mini` calls per run. Deep research makes the most.
- **SerpAPI:** 1 credit per flight search (**2 for a round trip**, because return flights are a second lookup) and 1 credit per hotel search.
- **Tavily:** 1 or more searches per web search or deep-research run.

Limits to keep in mind:

- Results are a snapshot from Google Flights / Google Hotels and the web. Always confirm prices and availability before booking. The system does **not** book anything.
- Return flights are listed for the cheapest outbound option only.
- Hotels without a price for your dates are left out.

### 1.8 Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENAI_API_KEY` / authentication errors | Check `.env` exists in the project folder and the key is valid. |
| Answer says flights or hotels couldn't be retrieved | Check `SERPAPI_API_KEY` and your SerpAPI credit balance. Make sure the dates are in the future. |
| `TAVILY_API_KEY environment variable not set` | Add the Tavily key to `.env`. |
| A follow-up question ignores the earlier conversation | Use the same `--thread-id` as before, and don't pass `--new`. Check with `--history`. |
| You want to delete all saved conversations | Delete `data/travel_planner_memory.sqlite` (this also removes the audit history). |

---

## Part 2 – Developer guide: setting up, updating and extending the system

### 2.1 Setting up your environment

The project is a standard Python package (`src/` layout) managed with **uv**. Dependency versions are pinned in `uv.lock`.

1. **Install uv** (if needed): `curl -LsSf https://astral.sh/uv/install.sh | sh`. See the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/) for other methods.
2. **Get the code** and open the project folder:
   ```bash
   cd smart-traveler
   ```
3. **Create the virtual environment and install everything:**
   ```bash
   uv sync
   ```
   This creates `.venv/` with Python 3.14 (pinned in `.python-version`; uv downloads it if missing). It installs the runtime dependencies from `pyproject.toml`, the `dev` group (pytest), and the `st` package itself in editable mode, so code changes take effect immediately.
4. **Add your API keys:** create `.env` as shown in [1.1 Before you start](#11-before-you-start). `.env.example` explains where to get each key.
5. **Check the setup:**
   ```bash
   uv run pytest            # 51 unit tests, no network calls, under a second
   uv run st --help
   ```
6. **Optional:** activate the environment with `source .venv/bin/activate` to use `python`, `pytest` and `st` directly. In VS Code, select `.venv/bin/python` as the interpreter.

<details>
<summary>Without uv (plain venv + pip)</summary>

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e . pytest
```

This installs from the version ranges in `pyproject.toml`, not the exact versions in `uv.lock`.
</details>

Managing dependencies:

```bash
uv add <package>          # add a runtime dependency (updates pyproject.toml and uv.lock)
uv add --dev <package>    # add a development-only dependency
uv lock --upgrade         # upgrade locked versions within the allowed ranges
```

### 2.2 Project layout

```
smart-traveler/
├── src/st/                     # the package (entry point: st.main:main)
│   ├── main.py                 # CLI
│   ├── config.py               # API key helper (not used at runtime)
│   ├── core/state.py           # shared graph state and data models
│   ├── agents/                 # orchestrator, search agent, planner agent, prompts
│   ├── tools/                  # flight, hotel, web search and deep research tools
│   ├── synthesizer/            # merges agent results into the final answer
│   ├── graph/builder.py        # wires everything into the LangGraph
│   └── memory/checkpointer.py  # multi-turn memory (SQLite checkpoints) and audit trail
├── tests/                      # pytest suite (unit tests + opt-in live test)
├── docs/                       # design notes and PlantUML architecture diagrams
├── docs/images/                # PlantUML source and rendered images
├── data/                       # runtime data, e.g. the memory database (git-ignored)
├── pyproject.toml / uv.lock    # dependencies and tool configuration
└── .env                        # your API keys (git-ignored, create it yourself)
```

### 2.3 How a request flows

1. **`main.py`** opens the memory database, builds the graph, and calls it with the user's query on a conversation thread.
2. **Orchestrator:** one structured-output LLM call decides which agents are needed and writes a **self-contained sub-query for each**. It resolves relative dates and follow-up references using the conversation so far. It also clears the previous turn's working state.
3. **Search agent** and/or **planner agent** run **in parallel**. Each loops: the LLM picks tool calls, its tool node runs them, and the LLM reads the results, until the agent produces its answer.
4. **Synthesizer** is a *deferred* node, so it runs once after every agent has finished. It merges their results into one answer.
5. After every step, the **checkpointer** saves the full state to SQLite. That is how memory and `--audit` work.

The [architecture diagrams](#26-architecture-diagrams) show each step in detail.

### 2.4 Modules

**`main.py`: command-line interface.** It parses the arguments described in [1.6](#16-cli-reference) and runs one of: a single turn (`run_travel_planning_workflow`), the chat loop (`chat`), `show_history` or `show_audit`. `create_travel_planner(checkpointer)` creates the shared LLM (`gpt-4o-mini`, temperature 0.7) and the graph.

**`core/state.py`: shared state and data models.**
- `TravelPlannerState` is the graph state that every node reads from and writes to:
  - `messages`: the conversation
  - `user_query`
  - `tasks`: the orchestrator's routing
  - `agent_results`
  - `final_answer`
  - `search_messages` / `planner_messages`: each agent's private working history
- **Reducers** control how writes combine. `add_messages` is used for the message lists. `append_or_reset` (append; `None` clears) is used for `agent_results`, so parallel agents can write at the same time and each turn can start clean.
- `AgentTask` is one agent's sub-query and focus. `ClassificationResult` is the orchestrator's structured output. `get_task(state, source)` lets an agent find its own task.

**`agents/orchestrator.py`: router.** `QueryClassifier.classify_query()` calls `ChatOpenAI(...).with_structured_output(ClassificationResult)` with a prompt that contains today's date and the last 6 conversation messages. It merges duplicate tasks per agent. If no task comes back, it falls back to the planner; if the call fails, it sends the full query to both agents. `classify_query_parallel(state)` is the graph node, and it also resets the per-turn channels.

**`agents/search_agent.py`: flight and hotel agent.** `create_search_agent(llm)` returns two nodes:
- `search_agent_node`: an LLM bound to `search_flights` and `search_hotels`
- `search_tool_node`: executes tool calls by name, and turns errors and unknown tools into `ToolMessage`s so the LLM can react

**`agents/planner_agent.py`: itinerary and Q&A agent.** Same structure as the search agent, with the `web_search` (quick facts) and `deep_research` (itineraries) tools.

**`agents/prompts.py`: all agent prompts.**
- `flightHotelSearch_instructions`: covers airport-code rules and round-trip presentation
- `travel_scout_instructions`: the planner's tool-selection rules, and it keeps the planner away from flights and hotels
- `research_instructions`: the deep-research sub-agent's prompt

**`tools/flight_search.py`: `search_flights` tool (SerpAPI Google Flights).**
- Validates IATA airport codes before spending a credit (comma-separated lists are allowed).
- Sends SerpAPI's parameter codes: `type` 1 or 2 for round trip or one-way, numeric `stops`.
- Trims the response to price insights and the top 8 options.
- For round trips, makes one extra lookup (using the `departure_token`) to fetch return flights for the top outbound option.

**`tools/hotel_search.py`: `search_hotels` tool (SerpAPI Google Hotels).** It returns the top 8 properties that have a price for the dates. `sort_by` can be `recommended`, `price_low`, `rating` or `most_reviewed`.

**`tools/web_search.py`: `web_search` tool (Tavily).** It returns Tavily's answer plus short snippets, without raw page content. It supports `search_depth`, `topic` and `time_range`.

**`tools/deep_research.py`: `deep_research` tool.**
- Runs a [deepagents](https://github.com/langchain-ai/deepagents) sub-agent that plans and researches with Tavily (advanced depth), then returns an itinerary with a rationale.
- The sub-agent is built once and cached; its model comes from `DEEP_RESEARCH_MODEL`.
- Each run is capped at 60 graph steps.

**`synthesizer/synthesizer.py`: final answer.** `create_synthesizer(llm)` returns a node that merges all `agent_results` into one structured answer. It includes only the sections the results cover, and it never invents prices.

**`graph/builder.py`: workflow wiring.** `build_parallel_travel_agent(llm, checkpointer=None)` adds the nodes and wires the routing:
- `route_to_agent` sends the request to one or both agents (both run in the same superstep).
- Each agent ⇄ tool loop repeats until the agent stops calling tools.
- The **deferred** synthesizer runs once, then the graph ends.

It compiles the graph with the optional checkpointer.

**`memory/checkpointer.py`: multi-turn memory and audit.**
- `open_checkpointer()` opens an `AsyncSqliteSaver` with a strict serializer allow-list. To move to Postgres, swap it for `AsyncPostgresSaver`.
- `thread_config()` selects a conversation thread.
- `new_turn_input()` builds the input for one turn.
- `get_conversation()` reads a thread's conversation.
- `get_audit_trail()` uses `aget_state_history` to list every node execution, with the tasks, tool calls and results it wrote.

**`config.py`: legacy API-key helper.** `APIConfig` validates the keys, but nothing at runtime uses it; the tools read keys with `os.getenv`. Importing the package runs `load_dotenv()` through this module.

**`tests/`: test suite.**
- `conftest.py`: shared stubs
- `test_deep_research.py`: the deep research tool
- `test_planner_agent.py`: planner tools and dispatch
- `test_search_tools.py`: flight, hotel and web search parameters and trimming
- `test_orchestration.py`: routing, sub-queries, deferred synthesis
- `test_memory.py`: persistence, per-turn resets, threads, audit, CLI
- `test_deep_research_live.py`: real API calls, opt-in

### 2.5 Testing

```bash
uv run pytest                          # unit tests: stubbed LLMs and APIs, no keys or network needed
uv run pytest tests/test_memory.py -v  # one file, verbose
uv run pytest -m live                  # live test: real OpenAI + Tavily calls (costs a few cents)
```

Live tests are marked `@pytest.mark.live` and are excluded by default (see `[tool.pytest.ini_options]` in `pyproject.toml`). For an end-to-end check, run a real query and inspect it:

```bash
uv run st --thread-id dev-check --new "Plan 2 days in Porto"
uv run st --audit --thread-id dev-check
```

### 2.6 Architecture diagrams

The `docs/images/` folder contains PlantUML diagrams of the system:

| File | Shows |
|---|---|
| `smart_travel_mas_high_level_overview.puml` | High-level component architecture: You → Orchestrator → Search/Planner agents → Synthesizer |
| `smart_travel_mas_e2e_dataflow.puml` | End-to-end dataflow including shared state, tools, and final recommendation |

To view them, install the VS Code **PlantUML** extension (needs Java and Graphviz) and press Alt+D, or render images with `java -jar plantuml.jar -tsvg docs/images/*.puml`. The main architectural documentation is in `docs/Building_a_Smart_Travel_Planning_MAS.md`, which walks through the design and implementation of this multi-agent system.

### 2.7 Common extensions

**Add a tool to an existing agent**
1. Create the tool in `src/st/tools/` with the `@tool` decorator. Read its API key with `os.getenv`, return compact text or JSON (not raw API payloads), and return errors as text.
2. Export it from `tools/__init__.py`.
3. Add it to the agent's tool list (`search_tools` in `search_agent.py` or `planner_tools` in `planner_agent.py`). Tool calls are dispatched by name automatically.
4. Tell the agent when to use it, in `agents/prompts.py`.
5. Add unit tests with a stubbed API client (see `tests/test_search_tools.py`).

**Add a new agent** (for example, a car-rental agent)
1. Create `agents/<name>_agent.py` following `search_agent.py`: an agent node that reads its task with `get_task(state, "<name>_agent")`, plus a tool node.
2. Add a `<name>_messages` channel to `TravelPlannerState`, and add `"<name>_agent"` to `AgentTask.source`.
3. Describe the agent and its routing rules in the orchestrator's `classification_prompt`. Clear its message channel in the orchestrator's per-turn reset.
4. In `graph/builder.py`: add the nodes, add the agent to `route_to_agent` and its `path_map`, add the tool → agent edge, and add an agent router that returns the tool node or `"synthesizer"`.
5. If you need to, mention its section in the synthesizer prompt. Then add tests (see `tests/test_orchestration.py`).

**Change models:** the shared agent and synthesizer LLM is created in `main.create_travel_planner()`. The orchestrator builds its own `ChatOpenAI(temperature=0)` in `agents/orchestrator.py`. The deep-research model is set with `DEEP_RESEARCH_MODEL`.

**Switch memory to Postgres:** `uv add langgraph-checkpoint-postgres`. Then in `memory/checkpointer.py`, change `open_checkpointer()` to open the connection with `async with AsyncPostgresSaver.from_conn_string(<postgres-url>) as checkpointer:`. Keep the strict serializer (`CHECKPOINT_TYPES`), call `await checkpointer.setup()` once to create the tables, and yield the saver. Callers only use the generic checkpointer API, so nothing else changes.

**Tune result sizes:** `MAX_FLIGHT_OPTIONS`, `MAX_RETURN_OPTIONS`, `MAX_HOTEL_OPTIONS` and `MAX_SNIPPET_CHARS` (tools); `MAX_HISTORY_MESSAGES` and `MAX_HISTORY_MESSAGE_CHARS` (orchestrator); `DEEP_RESEARCH_RECURSION_LIMIT` (deep research). Larger values give agents more context but cost more tokens.

---

## License

Copyright (C) 2026 Tonworio(Ton) Oguara. Licensed under AGPLv3.
