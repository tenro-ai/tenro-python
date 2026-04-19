# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Input/output extraction from spans for trace rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tenro._core.spans import AgentRun, BaseSpan, LLMCall, LLMScope


def get_input(span: BaseSpan) -> tuple[str, bool] | None:
    """Extract input text from span.

    Returns:
        Tuple of (text, is_quoted) or None.
    """
    from tenro._core.spans import AgentRun, LLMCall, LLMScope, ToolCall

    if isinstance(span, AgentRun):
        return _agent_input(span)
    if isinstance(span, LLMScope):
        return _scope_input(span)
    if isinstance(span, LLMCall):
        return _llm_call_input(span)
    if isinstance(span, ToolCall):
        return _kwargs_input(span.args, span.kwargs)
    return None


def get_output(span: BaseSpan) -> tuple[str, bool] | None:
    """Extract output text from span.

    Returns:
        Tuple of (text, is_quoted) or None.
    """
    from tenro._core.spans import AgentRun, LLMCall, LLMScope, ToolCall

    if isinstance(span, (AgentRun, LLMScope)):
        if span.output_data is not None:
            if isinstance(span.output_data, str):
                return (span.output_data, True)
            return (repr(span.output_data), False)
        return None
    if isinstance(span, LLMCall):
        return (span.response, True) if span.response else None
    if isinstance(span, ToolCall):
        if span.result is not None:
            if isinstance(span.result, str):
                return (span.result, True)
            return (repr(span.result), False)
        return None
    return None


def _agent_input(span: AgentRun) -> tuple[str, bool] | None:
    """Extract input from agent span, unwrapping single-element tuples."""
    if span.input_data is None:
        return None
    data = span.input_data
    if isinstance(data, tuple) and len(data) == 1:
        data = data[0]
    return (data, True) if isinstance(data, str) else (repr(data), False)


def _scope_input(span: LLMScope) -> tuple[str, bool] | None:
    """Extract input from LLM scope as formatted args/kwargs."""
    parts: list[str] = []
    if span.input_data:
        if len(span.input_data) == 1 and isinstance(span.input_data[0], str):
            parts.append(repr(span.input_data[0]))
        else:
            parts.extend(repr(a) for a in span.input_data)
    if span.input_kwargs:
        parts.extend(f"{k}={v!r}" for k, v in span.input_kwargs.items())
    return (", ".join(parts), False) if parts else None


def _llm_call_input(span: LLMCall) -> tuple[str, bool] | None:
    """Extract last message content from LLM call."""
    if not span.messages:
        return None
    content = span.messages[-1].get("content", "")
    return (str(content), True) if content else None


def _kwargs_input(
    args: tuple[object, ...] | None, kwargs: dict[str, object] | None
) -> tuple[str, bool] | None:
    """Format args/kwargs into a preview string."""
    parts: list[str] = []
    if args:
        parts.extend(repr(a) for a in args)
    if kwargs:
        parts.extend(f"{k}={v!r}" for k, v in kwargs.items())
    return (", ".join(parts), False) if parts else None


def count_spans(agents: list[AgentRun]) -> dict[str, int]:
    """Count spans by type across agent trees."""
    from tenro._core.spans import AgentRun, LLMCall, ToolCall

    counts = {"agents": 0, "llm_calls": 0, "tool_calls": 0}

    def count_recursive(agent: AgentRun) -> None:
        counts["agents"] += 1
        for span in agent.spans:
            if isinstance(span, AgentRun):
                count_recursive(span)
            elif isinstance(span, LLMCall):
                counts["llm_calls"] += 1
            elif isinstance(span, ToolCall):
                counts["tool_calls"] += 1

    for agent in agents:
        count_recursive(agent)

    return counts


def build_footer(agents: list[AgentRun]) -> list[str]:
    """Build the footer with summary statistics."""
    lines: list[str] = ["", "[dim]" + "\u2500" * 64 + "[/dim]"]

    stats = count_spans(agents)
    summary_parts = [
        f"{stats['agents']} agent{'s' if stats['agents'] != 1 else ''}",
        f"{stats['llm_calls']} LLM call{'s' if stats['llm_calls'] != 1 else ''}",
        f"{stats['tool_calls']} tool call{'s' if stats['tool_calls'] != 1 else ''}",
    ]

    if agents:
        total_ms = sum(a.latency_ms or 0.0 for a in agents)
        duration = f"{total_ms / 1000:.2f}s" if total_ms >= 1000 else f"{total_ms:.0f}ms"
        summary_parts.append(f"Total: {duration}")

    lines.append(f"[dim]Summary: {' | '.join(summary_parts)}[/dim]")
    lines.append("")
    return lines
