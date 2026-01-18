# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Common utilities for provider schema conversion.

Handles format detection, ID generation, and argument serialization.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from uuid_utils import uuid7


class ToolFormat(Enum):
    """Tool call input format types."""

    SIMPLIFIED = "simplified"  # ["tool1", "tool2"]
    MEDIUM = "medium"  # [{"name": "tool", "arguments": {...}}]
    FULL = "full"  # Provider-specific full format


def detect_tool_format(tools: list[str | dict[str, Any]], full_format_key: str) -> ToolFormat:
    """Detect tool call format from input.

    Args:
        tools: List of tool specifications.
        full_format_key: Key that indicates full format (e.g., "function" for OpenAI).

    Returns:
        Detected format type.

    Examples:
        >>> detect_tool_format(["tool1"], "function")
        ToolFormat.SIMPLIFIED
        >>> detect_tool_format([{"name": "tool"}], "function")
        ToolFormat.MEDIUM
        >>> detect_tool_format([{"function": {...}}], "function")
        ToolFormat.FULL
    """
    if not tools:
        return ToolFormat.SIMPLIFIED

    first_item = tools[0]

    # Simplified: list of strings
    if isinstance(first_item, str):
        return ToolFormat.SIMPLIFIED

    # Dict-based: check for full format key
    if isinstance(first_item, dict):
        if full_format_key in first_item:
            return ToolFormat.FULL
        return ToolFormat.MEDIUM

    # Default to simplified
    return ToolFormat.SIMPLIFIED


def generate_call_id(prefix: str = "call") -> str:
    """Generate unique tool call ID with provider-specific prefix.

    Args:
        prefix: ID prefix (e.g., "call" for OpenAI, "toolu" for Anthropic).

    Returns:
        Unique call ID in format: {prefix}_{24-char-hex}

    Examples:
        >>> generate_call_id("call")
        'call_abc123def456...'
        >>> generate_call_id("toolu")
        'toolu_xyz789...'
    """
    return f"{prefix}_{uuid7().hex[:24]}"


def serialize_arguments_to_json(arguments: Any) -> str:
    """Serialize arguments to JSON string (OpenAI format).

    Args:
        arguments: Arguments dict, string, or other type.

    Returns:
        JSON string representation.

    Examples:
        >>> serialize_arguments_to_json({"location": "Paris"})
        '{"location": "Paris"}'
        >>> serialize_arguments_to_json("{}")
        '{}'
    """
    if isinstance(arguments, dict):
        return json.dumps(arguments)
    elif isinstance(arguments, str):
        return arguments  # Already a string
    else:
        return json.dumps(arguments)


def serialize_arguments_to_dict(arguments: Any) -> dict[str, Any]:
    """Serialize arguments to dict (Anthropic/Gemini format).

    Args:
        arguments: Arguments dict, string, or other type.

    Returns:
        Dict representation (empty dict if not convertible).

    Examples:
        >>> serialize_arguments_to_dict({"location": "Paris"})
        {'location': 'Paris'}
        >>> serialize_arguments_to_dict("{}")
        {}
    """
    if isinstance(arguments, dict):
        return arguments
    elif isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            # Ensure it's a dict (json.loads can return other types)
            if isinstance(parsed, dict):
                return parsed
            return {}
        except (json.JSONDecodeError, ValueError):
            return {}
    else:
        return {}


def extract_tool_name_and_args(
    tool_dict: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Extract tool name and arguments from medium format dict.

    Args:
        tool_dict: Dict with "name" and optional "arguments" keys.

    Returns:
        Tuple of (tool_name, arguments_dict).

    Examples:
        >>> extract_tool_name_and_args({"name": "search", "arguments": {"q": "AI"}})
        ('search', {'q': 'AI'})
        >>> extract_tool_name_and_args({"name": "get_weather"})
        ('get_weather', {})
    """
    name = tool_dict.get("name", "")
    arguments = tool_dict.get("arguments", {})

    # Ensure arguments is a dict
    if not isinstance(arguments, dict):
        arguments = {}

    return name, arguments
