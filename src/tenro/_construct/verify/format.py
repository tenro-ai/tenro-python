# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Tree formatting utilities for verification error messages.

Formats span trees with emojis and tree connectors for clear error output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tenro.trace.icons import (
    ARROW_IN,
    ARROW_OUT,
    BRANCH,
    EMOJI_AGENT,
    EMOJI_LLM,
    EMOJI_TOOL,
    LAST,
    SPACE,
    VERTICAL,
)

if TYPE_CHECKING:
    from tenro._core.spans import BaseSpan


def get_span_emoji(span: BaseSpan) -> str:
    """Return emoji for span type.

    Args:
        span: The span to get emoji for.

    Returns:
        Emoji string for the span type.
    """
    from tenro._core.spans import AgentRun, LLMCall, ToolCall

    if isinstance(span, AgentRun):
        return EMOJI_AGENT
    if isinstance(span, LLMCall):
        return EMOJI_LLM
    if isinstance(span, ToolCall):
        return EMOJI_TOOL
    return ""


def get_span_name(span: BaseSpan) -> str:
    """Return display name for span.

    Args:
        span: The span to get name for.

    Returns:
        Display name string.
    """
    from tenro._core.spans import AgentRun, LLMCall, ToolCall

    if isinstance(span, AgentRun):
        return span.display_name or span.target_path
    if isinstance(span, LLMCall):
        provider = span.provider or "unknown"
        model = span.model or "unknown"
        return f"{provider}/{model}"
    if isinstance(span, ToolCall):
        return span.display_name or span.target_path
    return "unknown"


def get_span_input(span: BaseSpan) -> str:
    """Return input representation for span.

    Args:
        span: The span to get input for.

    Returns:
        Input string representation.
    """
    from tenro._core.spans import AgentRun, LLMCall, ToolCall

    if isinstance(span, AgentRun):
        if span.input_data is not None:
            return repr(span.input_data)
        if span.kwargs:
            return repr(span.kwargs)
        return ""
    if isinstance(span, LLMCall):
        if span.messages:
            last_msg = span.messages[-1]
            content = last_msg.get("content", "")
            if isinstance(content, str) and len(content) > 50:
                content = content[:47] + "..."
            return repr(content)
        return ""
    if isinstance(span, ToolCall):
        parts: list[str] = []
        if span.args:
            parts.extend(repr(a) for a in span.args)
        if span.kwargs:
            parts.extend(f"{k}={v!r}" for k, v in span.kwargs.items())
        return ", ".join(parts)
    return ""


def get_span_output(span: BaseSpan) -> str:
    """Return output representation for span.

    Args:
        span: The span to get output for.

    Returns:
        Output string representation.
    """
    from tenro._core.spans import AgentRun, LLMCall, ToolCall

    if isinstance(span, AgentRun):
        if span.output_data is not None:
            out = repr(span.output_data)
            if len(out) > 60:
                return out[:57] + "..."
            return out
        return ""
    if isinstance(span, LLMCall):
        if span.response:
            if len(span.response) > 50:
                return repr(span.response[:47] + "...")
            return repr(span.response)
        return ""
    if isinstance(span, ToolCall):
        if span.result is not None:
            out = repr(span.result)
            if len(out) > 60:
                return out[:57] + "..."
            return out
        return ""
    return ""


def format_span_tree(
    spans: list[BaseSpan],
    failed_span_id: str | None = None,
    indent: str = "",
) -> str:
    """Format spans as a tree with emojis.

    Args:
        spans: List of spans to format.
        failed_span_id: ID of the failed span to mark.
        indent: Current indentation prefix.

    Returns:
        Formatted tree string.
    """
    from tenro._core.spans import AgentRun

    lines: list[str] = []

    for i, span in enumerate(spans):
        is_last = i == len(spans) - 1
        connector = LAST if is_last else BRANCH
        child_indent = indent + (SPACE if is_last else f"{VERTICAL}  ")

        emoji = get_span_emoji(span)
        name = get_span_name(span)
        failed_marker = f"  {ARROW_OUT} FAILED" if span.id == failed_span_id else ""
        lines.append(f"{indent}{connector} {emoji} {name}{failed_marker}")

        span_input = get_span_input(span)
        if span_input:
            lines.append(f"{child_indent}{BRANCH} {ARROW_IN} {span_input}")

        span_output = get_span_output(span)
        if span_output:
            has_children = isinstance(span, AgentRun) and span.spans
            out_connector = BRANCH if has_children else LAST
            lines.append(f"{child_indent}{out_connector} {ARROW_OUT} {span_output}")

        if isinstance(span, AgentRun) and span.spans:
            lines.append(f"{child_indent}{VERTICAL}")
            child_tree = format_span_tree(span.spans, failed_span_id, child_indent)
            lines.append(child_tree)

    return "\n".join(lines)


def format_error_with_trace(
    header: str,
    expected: object,
    observed: object,
    span: BaseSpan | None = None,
) -> str:
    """Format verification error with optional trace context.

    Args:
        header: Error header line (e.g., "agent[0] output mismatch:").
        expected: Expected value.
        observed: Observed value.
        span: Optional span to show trace context for.

    Returns:
        Formatted error message string.
    """
    from tenro._core.spans import AgentRun

    lines = [
        header,
        f"  {BRANCH} {ARROW_IN} Expected: {expected!r}",
        f"  {LAST} {ARROW_OUT} Observed: {observed!r}",
    ]

    if isinstance(span, AgentRun) and span.spans:
        lines.append("")
        lines.append("Trace:")
        tree = format_span_tree([span], failed_span_id=span.id, indent="  ")
        lines.append(tree)

    return "\n".join(lines)


__all__ = [
    "format_error_with_trace",
    "format_span_tree",
    "get_span_emoji",
    "get_span_input",
    "get_span_name",
    "get_span_output",
]
