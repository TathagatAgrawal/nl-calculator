"""Builds the LangChain chat model pointed at OpenRouter's free API.

OpenRouter exposes an OpenAI-compatible endpoint, so we can reuse
`langchain_openai.ChatOpenAI` unmodified by just overriding `base_url` and
`api_key` — no custom HTTP/client code needed.
"""

import os

from langchain_openai import ChatOpenAI

# Free OpenRouter model with confirmed tool/function-calling support.
# Overridable via the OPENROUTER_MODEL env var without touching this code.
_DEFAULT_MODEL = "openai/gpt-oss-20b:free"


def build_chat_model() -> ChatOpenAI:
    """Construct a ChatOpenAI instance configured for OpenRouter.

    Raises:
        RuntimeError: if OPENROUTER_API_KEY isn't set, so failures happen
            immediately at startup rather than on the first query.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
    return ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
