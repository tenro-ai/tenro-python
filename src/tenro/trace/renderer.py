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
from tenro.trace.span_io import build_footer, get_input, get_output

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
        lines.extend(build_footer(agents))

        return "\n".join(lines)

    def _build_header(self, test_name: str | None) -> list[str]:
        """Build the header section."""
        title = f"Trace: {test_name}" if test_name else "Trace"
        return [f"\n[bold]{title}[/bold]", "[dim]" + "\u2500" * 64 + "[/dim]", ""]

    def _build_agent_tree(self, agent: AgentRun) -> list[str]:
        """Build tree for a single root agent."""
        return [self._format_span_header(agent), *self._build_span_content(agent, SPACE)]

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
        from tenro._core.spans import AgentRun, LLMScope, ToolCall

        lines: list[str] = []

        children: list[BaseSpan] = []
        if isinstance(span, AgentRun) and span.spans:
            children = self._reorganize_spans_for_display(span.spans)
        elif isinstance(span, (LLMScope, ToolCall)):
            children = getattr(span, "_render_children", [])

        input_result = get_input(span)
        output_result = get_output(span)
        has_children = bool(children)

        in_label = self._input_label(span)
        self._append_input_line(lines, indent, in_label, input_result, has_children, output_result)
        self._append_children(lines, indent, children, output_result, span.error)
        self._append_output_or_error(lines, indent, span, output_result, has_children)
        self._append_tool_requests(lines, indent, span)

        return lines

    def _input_label(self, span: BaseSpan) -> str:
        """Return semantic label for input based on span type."""
        from tenro._core.spans import AgentRun, LLMCall

        if isinstance(span, AgentRun):
            return "user: "
        if isinstance(span, LLMCall):
            return "prompt: "
        return ""

    def _append_input_line(
        self,
        lines: list[str],
        indent: str,
        label: str,
        input_result: tuple[str, bool] | None,
        has_children: bool,
        output_result: tuple[str, bool] | None,
    ) -> None:
        """Append formatted input line if present."""
        if input_result is None:
            return
        text, needs_quotes = input_result
        preview = self._truncate(text, self._config.max_preview_length)
        formatted = f'"{preview}"' if needs_quotes else preview
        connector = BRANCH if (has_children or output_result is not None) else LAST
        lines.append(f"{indent}{connector} {ARROW_IN} {label}{formatted}")

    def _append_children(
        self,
        lines: list[str],
        indent: str,
        children: list[BaseSpan],
        output_result: tuple[str, bool] | None,
        error: str | None,
    ) -> None:
        """Append child span tree if present."""
        if not children:
            return
        lines.append(f"{indent}{VERTICAL}")
        has_output = output_result is not None or error is not None
        lines.extend(self._build_tree(children, indent, has_output_after=has_output))

    def _append_output_or_error(
        self,
        lines: list[str],
        indent: str,
        span: BaseSpan,
        output_result: tuple[str, bool] | None,
        has_children: bool,
    ) -> None:
        """Append error or output line after children."""
        from tenro._core.spans import LLMCall

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
            lines.append(f"{indent}{connector} {ARROW_OUT} {formatted}")

    def _append_tool_requests(self, lines: list[str], indent: str, span: BaseSpan) -> None:
        """Append tool request line for LLM calls."""
        from tenro._core.spans import LLMCall

        if not (isinstance(span, LLMCall) and span.tool_calls):
            return
        tool_names = [tc.get("name", "?") for tc in span.tool_calls]
        lines.append(f"{indent}{LAST} 🔧 requests: {', '.join(tool_names)}")

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

        # Add [SIM] marker for simulated spans if enabled
        sim_marker = ""
        if self._config.show_simulation_marker and getattr(span, "simulated", False):
            sim_marker = " [dim]\\[SIM][/dim]"

        # Only show ERR on error, nothing on success
        if span.error:
            padding = " " * max(0, 55 - len(name))
            return f"{icon} [bold]{name}[/bold]{sim_marker}{padding} [bold red]ERR[/bold red]"
        return f"{icon} [bold]{name}[/bold]{sim_marker}"

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text with ellipsis."""
        clean = text.replace("\n", " ").replace("\r", "")
        if len(clean) <= max_length:
            return clean
        return clean[: max_length - 3] + "..."
