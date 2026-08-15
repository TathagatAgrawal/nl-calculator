# nl-calculator

A calculator that takes natural-language queries. An LLM parses the query, calls arithmetic tools to compute exact results, and replies in natural language.

```raw
[user]
"add 5 and 3, then multiply the result by 2"

[tool call] add(a=5, b=3) -> 8.0
[tool call] multiply(a=8, b=2) -> 16.0

[assistant]
"The sum of 5 and 3 is 8, and multiplying that by 2 gives 16."
```

## How it works

- **LLM & tool-calling**: [LangChain](https://python.langchain.com/) (`langchain-openai`'s `ChatOpenAI`) talks to [OpenRouter](https://openrouter.ai/)'s free-tier models over its OpenAI-compatible API. No API cost, no local model to run.
- **Tools**: `add`, `subtract`, `multiply`, `divide`, `power`, `sqrt` — plain Python functions wrapped as LangChain tools.
- **Validated arguments**: each tool's arguments are a Pydantic model (`args_schema`), so LangChain builds the tool's JSON schema from it and validates the model's arguments against that schema before the tool body ever runs (e.g. `sqrt` rejects negative inputs outright).
- **Multi-step queries**: the agent loop keeps calling tools and feeding results back to the model until it has a final answer, so chained requests like "add 5 and 3, then multiply by 2" resolve in one query.
- **Conversation memory**: each CLI session keeps its message history, so follow-ups like "now divide that by 2" refer back to the previous result.

## Project layout

```
nl_calculator/
  tools.py        # arithmetic functions, their Pydantic arg schemas, and TOOL_REGISTRY
  llm_client.py    # builds the LangChain ChatOpenAI client, pointed at OpenRouter
  agent.py         # Session: the tool-calling loop, independent of any UI
  cli.py           # terminal REPL built on top of Session
main.py             # entrypoint: `python main.py`
```

`agent.Session` has no CLI-specific logic, so a future web UI can reuse it directly without touching `tools.py`, `llm_client.py`, or `agent.py`.

## Setup

1. Get a free [OpenRouter](https://openrouter.ai/) API key (Settings → API Keys) — no payment info required for `:free` models.
2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY` (and optionally `OPENROUTER_MODEL`, if you want a different free model).

## Usage

```
venv\Scripts\python main.py
```

Type a math question at the `>` prompt (e.g. `what is 12 plus 7?`), or `exit`/`quit` to leave. Each tool call the model makes is printed before the final natural-language answer.

## Linting & formatting

```
venv\Scripts\python -m ruff format .
venv\Scripts\python -m ruff check .
```
