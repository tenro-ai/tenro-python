# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""LLM response types for simulation.

Provides structured response types for the `responses=` parameter:
- `LLMResponse`: Ordered blocks of text and tool calls
- `RawLLMResponse`: Raw provider JSON passthrough
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tenro.errors import TenroValidationError

if TYPE_CHECKING:
    from tenro.tool_calls import ToolCall as ToolCallType


def _get_tool_call_class() -> type:
    """Lazy import ToolCall to avoid circular imports."""
    from tenro.tool_calls import ToolCall

    return ToolCall


@dataclass
class LLMResponse:
    """Structured LLM response with ordered content blocks.

    Use this when you need interleaved text and tool calls in a single LLM turn.
    The outer `responses=` list controls how many LLM calls; blocks control what's
    in each call.

    Args:
        blocks: Ordered sequence of text strings and ToolCall objects.

    Examples:
        Single turn with interleaved content::

            # ONE LLM call returns text + tool call + more text
            responses=[
                LLMResponse(["Thinking", ToolCall(search, q="AI"), "Done"])
            ]

        Multiple turns (contrast - NOT interleaved)::

            # THREE separate LLM calls
            responses=["Thinking", ToolCall(search, q="AI"), "Done"]

        Multiple tool calls in one turn::

            responses=[LLMResponse([ToolCall(a), ToolCall(b)])]
    """

    blocks: list[str | ToolCallType]
    _explicit_tool_calls: bool = False

    def __post_init__(self) -> None:
        """Validate blocks field."""
        tool_call_cls = _get_tool_call_class()

        if not isinstance(self.blocks, list):
            raise TenroValidationError(f"blocks must be list, got {type(self.blocks).__name__}")
        for i, block in enumerate(self.blocks):
            if not isinstance(block, (str, tool_call_cls)):
                raise TenroValidationError(
                    f"blocks[{i}] must be str or ToolCall, got {type(block).__name__}"
                )

    @property
    def text(self) -> str | None:
        """Concatenated text blocks, or None if no text."""
        texts = [b for b in self.blocks if isinstance(b, str)]
        return "".join(texts) if texts else None

    @property
    def tool_calls(self) -> list[ToolCallType] | None:
        """Extract tool calls from blocks.

        Returns:
            None: No tool calls in response (text-only).
            []: Explicit empty list (tools available but none called).
            [ToolCall, ...]: List of tool calls.
        """
        tool_call_cls = _get_tool_call_class()
        tcs: list[ToolCallType] = [b for b in self.blocks if isinstance(b, tool_call_cls)]  # type: ignore[misc]
        if tcs:
            return tcs
        return [] if self._explicit_tool_calls else None

    def has_any_tool_calls(self) -> bool:
        """Check if response contains any ToolCall blocks."""
        tool_call_cls = _get_tool_call_class()
        return any(isinstance(b, tool_call_cls) for b in self.blocks)

    def tool_calls_field_present(self) -> bool:
        """Check if tool_calls field is present (for conflict detection).

        True if _explicit_tool_calls=True OR any ToolCall in blocks.
        Handles [[]] (explicit empty) correctly.
        """
        return self._explicit_tool_calls or self.has_any_tool_calls()


@dataclass
class RawLLMResponse:
    """Raw provider JSON passthrough.

    Use when you need to pass provider-specific JSON directly to the adapter
    without Tenro interpretation. The payload is passed through unchanged.

    Examples:
        >>> RawLLMResponse(payload={"choices": [{"message": {"content": "Hi"}}]})
        >>> RawLLMResponse(payload={"content": [{"type": "text", "text": "Hi"}]})
    """

    payload: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate payload field."""
        if not isinstance(self.payload, dict):
            raise TenroValidationError(f"payload must be dict, got {type(self.payload).__name__}")


__all__ = ["LLMResponse", "RawLLMResponse"]
