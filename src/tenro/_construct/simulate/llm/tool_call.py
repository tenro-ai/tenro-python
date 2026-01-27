# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Tool call normalization helpers for LLM simulation."""

from __future__ import annotations

import json
from typing import Any

from tenro._construct.simulate.llm.parser import ToolCallParser
from tenro.tool_calls import ToolCall, tc

_TOOL_SCHEMA_KEYS = {"description", "parameters", "json_schema", "strict", "input_schema"}


def _validate_not_tool_schema(d: dict[str, Any]) -> None:
    """Raise if dict looks like a tool schema, not a tool call."""
    schema_keys_found = _TOOL_SCHEMA_KEYS & d.keys()
    if schema_keys_found:
        raise ValueError(
            f"Dict appears to be a tool schema, not a tool call. "
            f"Found schema keys: {schema_keys_found}. "
            f"Tool calls have {{name, arguments}}."
        )
    if d.get("type") == "function" and "function" in d:
        func_obj = d["function"]
        if isinstance(func_obj, dict):
            func_schema_keys = _TOOL_SCHEMA_KEYS & func_obj.keys()
            if func_schema_keys:
                raise ValueError(
                    f"Dict appears to be an OpenAI tool schema wrapper "
                    f"(found {func_schema_keys} in function object). "
                    f"Tool calls have {{name, arguments}}."
                )


ToolCallInput = ToolCall | dict[str, Any] | str
"""Type alias for inputs accepted by normalize_tool_call()."""


def _validate_parsed_tool_call(name: Any, arguments: Any) -> tuple[str, dict[str, Any]]:
    """Validate name is str and arguments are JSON-serializable dict.

    Returns:
        Validated (name, arguments) tuple.
    """
    if not isinstance(name, str):
        raise TypeError(
            f"Dict tool call 'name' must be str, got {type(name).__name__}. "
            f"For callables, use ToolCall(callable, ...)."
        )

    if not isinstance(arguments, dict):
        raise TypeError(f"Tool call 'arguments' must be dict, got {type(arguments).__name__}")

    try:
        json.dumps(arguments, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"Tool call arguments are not JSON-serializable: {e}. "
            f"Dict inputs must be JSON-safe; use ToolCall() for automatic coercions."
        ) from e

    return name, arguments


def normalize_tool_call(input_val: ToolCallInput) -> ToolCall:
    """Normalize various input formats to ToolCall.

    Accepts:
    - ToolCall: passthrough
    - str: name only, arguments={}
    - dict: {"name": "...", "arguments": {...}} or OpenAI/Anthropic formats

    Args:
        input_val: Tool call in any supported format.

    Returns:
        Normalized ToolCall instance.

    Raises:
        ValueError: If dict looks like a tool schema.
        TypeError: If input cannot be converted.
    """
    if isinstance(input_val, ToolCall):
        return input_val

    if isinstance(input_val, str):
        return ToolCall(name=input_val)

    if isinstance(input_val, dict):
        _validate_not_tool_schema(input_val)
        name, arguments, call_id = ToolCallParser.parse(input_val)

        name, arguments = _validate_parsed_tool_call(name, arguments)
        return ToolCall(name=name, arguments=arguments, call_id=call_id)

    raise TypeError(f"Cannot convert {type(input_val).__name__} to ToolCall")


def normalize_tool_calls(
    inputs: list[ToolCallInput] | None,
) -> list[ToolCall]:
    """Normalize list of tool call inputs.

    Args:
        inputs: List of tool calls in various formats, or None.

    Returns:
        List of normalized ToolCall instances.
    """
    if inputs is None:
        return []
    return [normalize_tool_call(i) for i in inputs]


__all__ = [
    "ToolCall",
    "ToolCallInput",
    "normalize_tool_call",
    "normalize_tool_calls",
    "tc",
]
