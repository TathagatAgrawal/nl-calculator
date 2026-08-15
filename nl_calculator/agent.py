"""Core LLM tool-calling loop, built on LangChain, independent of any UI.

The loop is hand-written (rather than a LangChain AgentExecutor) because we
need multi-turn conversation history that persists across separate
`run_query` calls (so follow-ups like "now divide that by 2" work), and a
per-tool-call callback for the CLI to print what's happening. LangChain
still does the heavy lifting: the chat model wrapper, message types, tool
binding, and — importantly — validating each tool call's arguments against
its Pydantic `args_schema` before the tool body ever runs.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from . import llm_client
from .tools import TOOL_REGISTRY, TOOLS

SYSTEM_PROMPT = (
    "You are a calculator assistant. For any arithmetic question, use the "
    "provided tools to compute the exact result instead of calculating it "
    "yourself. Then explain the result in a short, natural-language reply. "
    "If a tool call fails, explain the error clearly to the user."
)

# Safety cap on how many tool-call round-trips a single query can take,
# so a model stuck in a tool-calling loop can't hang the session forever.
MAX_TOOL_ROUNDS = 5


def _run_tool_call(name: str, args: dict) -> str:
    """Look up and invoke a tool by name, returning a JSON-serialized result.

    `tool.invoke(args)` re-validates `args` against the tool's Pydantic
    `args_schema`. Both schema-validation failures and business-logic
    errors raised by the tool body (e.g. divide-by-zero) are caught here
    and turned into a string the model can read and explain to the user,
    rather than crashing the session.
    """
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'."

    try:
        result = tool.invoke(args)
    except ValidationError as exc:
        return f"Error: invalid arguments for {name}: {exc}"
    except Exception as exc:  # noqa: BLE001 - relay any tool failure to the model
        return f"Error: {exc}"

    return json.dumps(result)


class Session:
    """Holds LangChain conversation history so follow-up queries work."""

    def __init__(self):
        # bind_tools attaches each tool's JSON schema (derived from its
        # Pydantic args_schema) to every request, so the model knows what
        # it can call and with what argument shape.
        self.llm = llm_client.build_chat_model().bind_tools(TOOLS)
        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]

    def run_query(self, user_text: str, on_tool_call=None) -> str:
        """Answer one user query, running any tool calls the model requests.

        Args:
            user_text: the user's natural-language question.
            on_tool_call: optional callback `(name, args, result_str)`
                invoked each time a tool runs, so a UI can display it.

        Returns:
            The model's final natural-language reply.
        """
        self.messages.append(HumanMessage(content=user_text))

        for _ in range(MAX_TOOL_ROUNDS):
            ai_message: AIMessage = self.llm.invoke(self.messages)
            self.messages.append(ai_message)

            if not ai_message.tool_calls:
                return ai_message.content or ""

            for tool_call in ai_message.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]
                result = _run_tool_call(name, args)

                if on_tool_call is not None:
                    on_tool_call(name, args, result)

                self.messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

        return "Sorry, I couldn't finish computing that."


def run_query(user_text: str) -> str:
    """One-off convenience call for a fresh session (no conversation history)."""
    return Session().run_query(user_text)
