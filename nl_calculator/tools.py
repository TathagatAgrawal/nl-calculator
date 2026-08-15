"""Arithmetic tools exposed to the LLM.

Each tool's arguments are defined as a Pydantic model (`args_schema`) rather
than plain function type hints. LangChain turns that schema into the JSON
schema sent to the model for tool/function calling, and validates the
model's returned arguments against it before our code ever sees them. That
means malformed or out-of-range arguments (e.g. a negative number passed to
`sqrt`) are rejected with a clear `ValidationError` instead of silently
misbehaving or crashing deeper in the call stack.
"""

import math

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class TwoNumberInput(BaseModel):
    """Shared argument schema for binary arithmetic operations."""

    a: float = Field(description="The first number.")
    b: float = Field(description="The second number.")


class PowerInput(BaseModel):
    """Argument schema for exponentiation."""

    base: float = Field(description="The base number.")
    exponent: float = Field(description="The exponent to raise the base to.")


class SqrtInput(BaseModel):
    """Argument schema for square root.

    `ge=0` makes negative inputs fail Pydantic validation before our
    function body runs, so the model gets immediate, structured feedback
    instead of a runtime exception.
    """

    a: float = Field(ge=0, description="The non-negative number to take the square root of.")


@tool("add", args_schema=TwoNumberInput)
def add(a: float, b: float) -> float:
    """Add two numbers together and return the sum."""
    return a + b


@tool("subtract", args_schema=TwoNumberInput)
def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the difference."""
    return a - b


@tool("multiply", args_schema=TwoNumberInput)
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together and return the product."""
    return a * b


@tool("divide", args_schema=TwoNumberInput)
def divide(a: float, b: float) -> float:
    """Divide a by b and return the quotient.

    Division by zero can't be expressed as a Pydantic field constraint (it
    depends on the *other* field's value), so it's still checked here at
    call time and surfaced as a normal exception for the agent loop to
    catch and relay to the model.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool("power", args_schema=PowerInput)
def power(base: float, exponent: float) -> float:
    """Raise base to the given exponent and return the result."""
    return base**exponent


@tool("sqrt", args_schema=SqrtInput)
def sqrt(a: float) -> float:
    """Return the square root of a (a is guaranteed non-negative by SqrtInput)."""
    return math.sqrt(a)


# All tools handed to the LLM via `.bind_tools(TOOLS)`.
TOOLS = [add, subtract, multiply, divide, power, sqrt]

# name -> LangChain tool, used to dispatch a model's tool call by name.
TOOL_REGISTRY = {t.name: t for t in TOOLS}
