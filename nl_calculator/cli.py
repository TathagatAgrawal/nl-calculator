"""Command-line REPL for the natural-language calculator.

This module only talks to `agent.Session`'s public interface, so it can be
swapped for a web UI later without touching `tools.py`, `llm_client.py`, or
`agent.py`.
"""

from dotenv import load_dotenv

from .agent import Session


def _print_tool_call(name: str, args: dict, result: str) -> None:
    """Render a single tool call as a readable line, e.g. `add(a=5, b=3) -> 8`."""
    args_str = ", ".join(f"{k}={v}" for k, v in args.items())
    print(f"  [tool call] {name}({args_str}) -> {result}")


def main() -> None:
    """Run the interactive REPL loop until the user exits."""
    load_dotenv()  # loads OPENROUTER_API_KEY / OPENROUTER_MODEL from .env
    session = Session()
    print("NL Calculator — type a math question, or 'exit' to quit.")

    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        try:
            reply = session.run_query(user_text, on_tool_call=_print_tool_call)
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive on errors
            reply = f"Error: {exc}"

        print(reply)


if __name__ == "__main__":
    main()
