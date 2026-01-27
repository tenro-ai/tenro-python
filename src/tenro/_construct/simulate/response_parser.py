# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Response parser for LLM simulation.

Handles type discrimination for the `responses=` parameter, converting
various input formats to LLMResponse or RawLLMResponse.
"""

from __future__ import annotations

from typing import Any

from tenro.llm_response import LLMResponse, RawLLMResponse
from tenro.tool_calls import ToolCall

ResponseItem = (
    str | Exception | ToolCall | list[Any] | LLMResponse | RawLLMResponse | dict[str, Any]
)
ParsedResponse = LLMResponse | Exception | RawLLMResponse


def parse_response_item(item: ResponseItem) -> ParsedResponse:
    """Parse a single response item into LLMResponse or RawLLMResponse.

    Normalizes any response format:
    - "Hello"              → text response
    - ToolCall(fn, x=1)    → single tool call
    - [ToolCall(...), ...] → multiple tool calls in one turn
    - ["Think", ToolCall()] → interleaved text and tool calls
    - LLMResponse(...)     → explicit block control
    - RawLLMResponse(...)  → raw provider JSON (no processing)
    - Exception            → simulates API error
    - {"text": ...}        → legacy format (deprecated)

    Args:
        item: Response item in any supported format.

    Returns:
        Parsed response as LLMResponse, RawLLMResponse, or Exception.

    Raises:
        TypeError: If item type is not supported or contents are invalid.
    """
    if isinstance(item, str):
        return LLMResponse(blocks=[item])

    if isinstance(item, (Exception, LLMResponse, RawLLMResponse)):
        return item

    if isinstance(item, ToolCall):
        return LLMResponse(blocks=[item])

    if isinstance(item, list):
        return _parse_list_item(item)

    if isinstance(item, dict):
        return _parse_dict_item(item)

    raise TypeError(
        f"Response item must be str, Exception, ToolCall, list, LLMResponse, "
        f"RawLLMResponse, or dict, got {type(item).__name__}"
    )


def _parse_list_item(items: list[Any]) -> LLMResponse:
    """Parse list response item.

    List is a "tool-related shorthand" and must be:
    - Empty ([]) → explicit empty tool_calls
    - Contains at least one ToolCall → tool calls (possibly with interleaved text)

    All-string lists are NOT allowed (use LLMResponse for multiple text blocks).
    This keeps "list == tool-related" mental model crisp.

    Args:
        items: List of items to parse.

    Returns:
        LLMResponse with blocks.

    Raises:
        TypeError: If list contents are invalid.
    """
    if not items:
        return LLMResponse(blocks=[], _explicit_tool_calls=True)

    blocks: list[str | ToolCall] = []
    has_tool_call = False

    for i, item in enumerate(items):
        if isinstance(item, ToolCall):
            blocks.append(item)
            has_tool_call = True
        elif isinstance(item, str):
            blocks.append(item)
        elif isinstance(item, list):
            raise TypeError(
                "Nested lists not allowed in responses. "
                "Use a flat list: [ToolCall(...), ToolCall(...)]"
            )
        else:
            raise TypeError(
                f"List items must be str or ToolCall, got {type(item).__name__} at index {i}"
            )

    if not has_tool_call:
        raise TypeError(
            "List must contain at least one ToolCall. "
            "For multiple text blocks, use LLMResponse(blocks=[...])."
        )

    return LLMResponse(blocks=blocks, _explicit_tool_calls=True)


def _parse_dict_item(d: dict[str, Any]) -> LLMResponse:
    """Parse dict response item as legacy format {text, tool_calls}.

    All dicts are treated as legacy format. For raw provider JSON,
    use RawLLMResponse(payload={...}) explicitly.

    Requires at least one of 'text' or 'tool_calls' keys (empty dict is ambiguous).
    Unknown keys are ignored (permissive parsing for backward compat).

    Args:
        d: Dict to parse.

    Returns:
        LLMResponse with blocks.

    Raises:
        TypeError: If dict contents are invalid or missing required keys.
    """
    from tenro._construct.simulate.llm.tool_call import normalize_tool_call

    has_text = "text" in d
    has_tool_calls = "tool_calls" in d

    if not has_text and not has_tool_calls:
        raise TypeError(
            "Dict responses must have 'text' or 'tool_calls'. "
            "For raw provider JSON, use RawLLMResponse(payload=...)."
        )

    blocks: list[str | ToolCall] = []

    text = d.get("text")
    if text is not None:
        if not isinstance(text, str):
            raise TypeError(f"dict 'text' must be str, got {type(text).__name__}")
        blocks.append(text)

    tool_calls = d.get("tool_calls")
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            raise TypeError(f"dict 'tool_calls' must be list, got {type(tool_calls).__name__}")
        for tc in tool_calls:
            if isinstance(tc, ToolCall):
                blocks.append(tc)
            else:
                # Legacy format supports dict/str tool calls
                blocks.append(normalize_tool_call(tc))

    return LLMResponse(blocks=blocks, _explicit_tool_calls=has_tool_calls)


def contributes_tool_calls(item: ResponseItem) -> bool:
    """Check if response item contributes tool_calls (for conflict detection).

    Uses "tool_calls field present" semantics, not just "any ToolCall exists".
    This correctly handles [[]] (explicit empty) as a conflict.

    Returns True for:
    - ToolCall (always)
    - list with any ToolCall OR empty list (tool-related shorthand)
    - LLMResponse with _explicit_tool_calls=True OR any ToolCall in blocks
    - dict containing "tool_calls" key (legacy format)

    Returns False for:
    - RawLLMResponse (passthrough, no conflict detection)
    - str, Exception

    Args:
        item: Response item to check.

    Returns:
        True if item contributes tool_calls.
    """
    if isinstance(item, (ToolCall, list)):
        return True
    if isinstance(item, LLMResponse):
        return item.tool_calls_field_present()
    if isinstance(item, dict):
        return "tool_calls" in item
    # str, Exception, RawLLMResponse don't contribute tool_calls
    return False


__all__ = [
    "ParsedResponse",
    "ResponseItem",
    "contributes_tool_calls",
    "parse_response_item",
]
