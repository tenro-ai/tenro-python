# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Trace renderer for console output.

Main TraceRenderer class that coordinates extraction and formatting
to produce trace visualization output.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from tenro.trace.icons import (
    ARROW_IN,
    ARROW_OUT,
    BRANCH,
    LAST,
    SPACE,
    VERTICAL,
    get_span_icon,
)

if TYPE_CHECKING:
    from tenro._core.spans import AgentRun, BaseSpan
    from tenro.trace.config import TraceConfig


class TraceRenderer:
    """Renders trace visualization to console.

    Coordinates extraction of span data and Rich formatting to produce
    a timeline view of the trace.

    Example:
        >>> from tenro.trace import TraceRenderer
        >>> renderer = TraceRenderer()
        >>> renderer.render(construct.agent_runs)
    """

    def __init__(self, config: TraceConfig | None = None) -> None:
        """Initialize renderer with configuration.

        Args:
            config: Trace configuration. If None, reads from environment.
        """
        from tenro.trace.config import get_trace_config

        self._config = config or get_trace_config()

    def render(self, agents: list[AgentRun], test_name: str | None = None) -> None:
        """Render agent traces to console.

        Args:
            agents: List of root agent runs to render.
            test_name: Optional test name for the header.
        """
        if not agents:
            return

        output = self.render_to_string(agents, test_name)

        from rich.console import Console

        console = Console()
        console.print(output)

    def render_to_string(self, agents: list[AgentRun], test_name: str | None = None) -> str:
        """Render traces to string for testing.

        Args:
            agents: List of root agent runs to render.
            test_name: Optional test name for the header.

        Returns:
            Formatted trace output as string.
        """
        if not agents:
            return ""

        lines: list[str] = []

        # Header
        lines.extend(self._build_header(test_name))

        # Root agents (no tree prefix, separated by blank lines)
        for i, agent in enumerate(agents):
            if i > 0:
                lines.append("")  # Blank line between root agents
            lines.extend(self._build_agent_tree(agent))

        # Footer with summary
        lines.extend(self._build_footer(agents))

        return "\n".join(lines)

    def _build_header(self, test_name: str | None) -> list[str]:
        """Build the header section."""
        lines: list[str] = []
        title = f"Trace: {test_name}" if test_name else "Trace"
        lines.append(f"\n[bold]{title}[/bold]")
        lines.append("[dim]" + "\u2500" * 64 + "[/dim]")
        lines.append("")
        return lines

    def _build_agent_tree(self, agent: AgentRun) -> list[str]:
        """Build tree for a single root agent."""
        lines: list[str] = []

        lines.append(self._format_span_header(agent))

        lines.extend(self._build_span_content(agent, SPACE))

        return lines

    def _build_tree(
        self, spans: Sequence[BaseSpan], indent: str, has_output_after: bool = False
    ) -> list[str]:
        """Build tree representation of child spans recursively."""
        lines: list[str] = []

        for i, span in enumerate(spans):
            is_last = i == len(spans) - 1 and not has_output_after
            connector = LAST if is_last else BRANCH
            child_indent = indent + (SPACE if is_last else f"{VERTICAL}  ")

            lines.append(f"{indent}{connector} {self._format_span_header(span)}")
            lines.extend(self._build_span_content(span, child_indent))

        return lines

    def _reorganize_spans_for_display(self, spans: Sequence[BaseSpan]) -> list[BaseSpan]:
        """Reorganize spans for visual nesting.

        Data model keeps spans as siblings, but for display this groups:
        - LLMCalls under their LLMScope (via llm_scope_id)
        - AgentRuns under their invoking ToolCall (via invoked_by_tool_call_id)
        """
        from tenro._core.spans import AgentRun, LLMCall, LLMScope, ToolCall

        scopes: dict[str, LLMScope] = {}
        tools: dict[str, ToolCall] = {}
        for span in spans:
            if isinstance(span, LLMScope):
                scopes[span.span_id] = span
                span._render_children = []  # type: ignore[attr-defined]
            elif isinstance(span, ToolCall):
                tools[span.span_id] = span
                span._render_children = []  # type: ignore[attr-defined]

        result: list[BaseSpan] = []
        for span in spans:
            if isinstance(span, LLMCall) and span.llm_scope_id in scopes:
                scopes[span.llm_scope_id]._render_children.append(span)  # type: ignore[attr-defined]
            elif isinstance(span, AgentRun) and span.invoked_by_tool_call_id in tools:
                tools[span.invoked_by_tool_call_id]._render_children.append(span)  # type: ignore[attr-defined]
            else:
                result.append(span)

        return result

    def _build_span_content(self, span: BaseSpan, indent: str) -> list[str]:
        """Build input, children, and output for any span type."""
        from tenro._core.spans import AgentRun, LLMCall, LLMScope, ToolCall

        lines: list[str] = []

        children: list[BaseSpan] = []
        if isinstance(span, AgentRun) and span.spans:
            children = self._reorganize_spans_for_display(span.spans)
        elif isinstance(span, (LLMScope, ToolCall)):
            # LLMScope/ToolCall may have virtual children attached by reorganize
            children = getattr(span, "_render_children", [])

        if not self._config.show_io_preview:
            if children:
                lines.append(f"{indent}{VERTICAL}")
                lines.extend(self._build_tree(children, indent))
            return lines

        input_result = self._get_input(span)
        output_result = self._get_output(span)
        has_children = bool(children)

        # Determine semantic labels based on span type
        if isinstance(span, AgentRun):
            in_label = "user: "
            out_label = ""
        elif isinstance(span, LLMCall):
            in_label = "prompt: "
            out_label = ""
        elif isinstance(span, ToolCall):
            in_label = ""
            out_label = ""
        else:
            in_label = ""
            out_label = ""

        # Input (always before children)
        if input_result is not None:
            text, needs_quotes = input_result
            preview = self._truncate(text, self._config.max_preview_length)
            formatted = f'"{preview}"' if needs_quotes else preview
            connector = BRANCH if (has_children or output_result is not None) else LAST
            lines.append(f"{indent}{connector} {ARROW_IN} {in_label}{formatted}")

        # Children (agents have spans, LLMScopes have virtual children)
        if children:
            lines.append(f"{indent}{VERTICAL}")
            has_output = output_result is not None or span.error is not None
            lines.extend(self._build_tree(children, indent, has_output_after=has_output))

        # Error or Output (always after children)
        # For LLMCall, check if tool_calls exist to determine final connector
        has_tool_calls = isinstance(span, LLMCall) and span.tool_calls

        if span.error:
            if has_children:
                lines.append(f"{indent}{VERTICAL}")
            connector = BRANCH if has_tool_calls else LAST
            lines.append(f"{indent}{connector} [red]{ARROW_OUT} error: {span.error}[/red]")
        elif output_result is not None:
            text, needs_quotes = output_result
            preview = self._truncate(text, self._config.max_preview_length)
            formatted = f'"{preview}"' if needs_quotes else preview
            if has_children:
                lines.append(f"{indent}{VERTICAL}")
            connector = BRANCH if has_tool_calls else LAST
            lines.append(f"{indent}{connector} {ARROW_OUT} {out_label}{formatted}")

        # Show tools the LLM requested
        if has_tool_calls:
            assert isinstance(span, LLMCall)  # for type narrowing
            tool_names = [tc.get("name", "?") for tc in span.tool_calls]  # type: ignore[union-attr]
            tools_str = ", ".join(tool_names)
            lines.append(f"{indent}{LAST} 🔧 requests: {tools_str}")

        return lines

    def _format_span_header(self, span: BaseSpan) -> str:
        """Format span header with icon, name, status."""
        from tenro._core.spans import AgentRun, LLMCall, LLMScope, ToolCall

        if isinstance(span, AgentRun):
            icon = get_span_icon("AGENT")
            name = span.display_name or span.target_path
        elif isinstance(span, LLMScope):
            icon = get_span_icon("LLM_SCOPE")
            name = span.caller_name or "llm_scope"
        elif isinstance(span, LLMCall):
            icon = get_span_icon("LLM")
            # Just model name, no provider prefix
            name = span.model or "unknown"
        elif isinstance(span, ToolCall):
            icon = get_span_icon("TOOL")
            name = span.display_name or span.target_path
        else:
            icon = "?"
            name = "unknown"

        # Only show ERR on error, nothing on success
        if span.error:
            padding = " " * max(0, 55 - len(name))
            return f"{icon} [bold]{name}[/bold]{padding} [bold red]ERR[/bold red]"
        return f"{icon} [bold]{name}[/bold]"

    def _get_input(self, span: BaseSpan) -> tuple[str, bool] | None:
        """Extract input text from span.

        Args:
            span: The span to extract input from.

        Returns:
            Tuple of (text, is_quoted) or None. is_quoted indicates if quotes needed.
        """
        from tenro._core.spans import AgentRun, LLMCall, LLMScope, ToolCall

        if isinstance(span, AgentRun):
            if span.input_data is not None:
                data = span.input_data
                # Unwrap single-element tuples (common from *args capture)
                if isinstance(data, tuple) and len(data) == 1:
                    data = data[0]
                if isinstance(data, str):
                    return (data, True)
                return (repr(data), False)
            return None
        elif isinstance(span, LLMScope):
            # LLMScope: format as (args, kwargs)
            parts: list[str] = []
            if span.input_data:
                if len(span.input_data) == 1 and isinstance(span.input_data[0], str):
                    parts.append(repr(span.input_data[0]))
                else:
                    parts.extend(repr(a) for a in span.input_data)
            if span.input_kwargs:
                parts.extend(f"{k}={v!r}" for k, v in span.input_kwargs.items())
            return (", ".join(parts), False) if parts else None
        elif isinstance(span, LLMCall):
            if span.messages:
                last_msg = span.messages[-1]
                content = last_msg.get("content", "")
                return (str(content), True) if content else None
            return None
        elif isinstance(span, ToolCall):
            parts = []
            if span.args:
                parts.extend(repr(a) for a in span.args)
            if span.kwargs:
                parts.extend(f"{k}={v!r}" for k, v in span.kwargs.items())
            return (", ".join(parts), False) if parts else None
        return None

    def _get_output(self, span: BaseSpan) -> tuple[str, bool] | None:
        """Extract output text from span.

        Args:
            span: The span to extract output from.

        Returns:
            Tuple of (text, is_quoted) or None. is_quoted indicates if quotes needed.
        """
        from tenro._core.spans import AgentRun, LLMCall, LLMScope, ToolCall

        if isinstance(span, (AgentRun, LLMScope)):
            if span.output_data is not None:
                if isinstance(span.output_data, str):
                    return (span.output_data, True)
                return (repr(span.output_data), False)
            return None
        elif isinstance(span, LLMCall):
            if span.response:
                return (span.response, True)
            return None
        elif isinstance(span, ToolCall):
            if span.result is not None:
                if isinstance(span.result, str):
                    return (span.result, True)
                return (repr(span.result), False)
            return None
        return None

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text with ellipsis."""
        clean = text.replace("\n", " ").replace("\r", "")
        if len(clean) <= max_length:
            return clean
        return clean[: max_length - 3] + "..."

    def _build_footer(self, agents: list[AgentRun]) -> list[str]:
        """Build the footer with summary statistics."""
        lines: list[str] = []
        lines.append("")
        lines.append("[dim]" + "\u2500" * 64 + "[/dim]")

        stats = self._count_spans(agents)
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

    def _count_spans(self, agents: list[AgentRun]) -> dict[str, int]:
        """Count spans by type."""
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
