# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Tool verification methods for construct testing.

Provides 4 core verification methods for tool calls:
- verify_tool() - flexible verification with argument matching
- verify_tool_never() - explicit "never called" check
- verify_tool_sequence() - order verification
- verify_tools() - aggregate/range queries
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tenro._construct.verify.engine import (
    verify_call_count,
    verify_sequence,
    verify_with_arguments,
)
from tenro._construct.verify.output import (
    verify_output,
    verify_output_contains,
    verify_output_exact,
)

if TYPE_CHECKING:
    from tenro._core.spans import AgentRun, ToolCall


class ToolVerifications:
    """Tool verification methods."""

    def __init__(self, tool_calls: list[ToolCall], agent_runs: list[AgentRun]) -> None:
        """Initialize with tool calls and agent runs.

        Args:
            tool_calls: List of tool call spans.
            agent_runs: List of agent run spans.
        """
        self._tool_calls = tool_calls
        self._agent_runs = agent_runs

    def verify_tool(
        self,
        target: str | tuple[str, ...],
        args_dict: dict[str, Any] | None = None,
        *,
        output: Any = None,
        output_contains: str | None = None,
        output_exact: Any = None,
        call_index: int | None = 0,
        **kwargs: Any,
    ) -> None:
        """Verify tool was called with optional argument and output matching.

        Args:
            target: Name or path of the tool. Can be a tuple of paths to match
                ANY of multiple entry method paths (for class targets).
            args_dict: Dict of expected arguments.
            output: Expected call response (dict=subset match, scalar=exact).
            output_contains: Expected substring in call response (strings only).
            output_exact: Expected call response (strict deep equality).
            call_index: Which call to check (0=first, -1=last, None=any).
            **kwargs: Expected keyword arguments.

        Raises:
            AssertionError: If verification fails.

        Examples:
            >>> construct.verify_tool(fetch_data)  # at least once
            >>> construct.verify_tool(fetch_data, user_id=123)  # with arg
            >>> construct.verify_tool(get_weather, output={"temp": 72})  # subset
            >>> construct.verify_tool(search, output_contains="results")  # substring
            >>> construct.verify_tool(api, output_exact={"a": 1, "b": 2})  # strict
        """
        targets = (target,) if isinstance(target, str) else target

        matching_calls = [c for c in self._tool_calls if c.target_path in targets]

        # Output verification (takes precedence)
        if output is not None:
            verify_output(matching_calls, "result", output, "tool", call_index=call_index)
            return
        if output_contains is not None:
            verify_output_contains(
                matching_calls, "result", output_contains, "tool", call_index=call_index
            )
            return
        if output_exact is not None:
            verify_output_exact(
                matching_calls, "result", output_exact, "tool", call_index=call_index
            )
            return

        # Input/count verification
        if not args_dict and not kwargs:
            verify_call_count(
                calls=self._tool_calls,
                agent_runs=self._agent_runs,
                count=None,
                min=1,
                max=None,
                name_filter=target,
                agent_filter=None,
                event_type="tool",
            )
        else:
            verify_with_arguments(
                calls=self._tool_calls,
                agent_runs=self._agent_runs,
                name=target,
                called_with=args_dict,
                agent_filter=None,
                event_type="tool",
                kwargs=kwargs,
            )

    def verify_tool_never(self, target: str | tuple[str, ...]) -> None:
        """Verify tool was never called.

        Args:
            target: Name or path of the tool. Can be a tuple of paths.

        Raises:
            AssertionError: If tool was called.

        Examples:
            >>> construct.verify_tool_never("dangerous_operation")
        """
        verify_call_count(
            calls=self._tool_calls,
            agent_runs=self._agent_runs,
            count=0,
            min=None,
            max=None,
            name_filter=target,
            agent_filter=None,
            event_type="tool",
        )

    def verify_tool_sequence(self, expected_sequence: list[str]) -> None:
        """Verify tools were called in a specific order.

        Args:
            expected_sequence: Expected sequence of tool names.

        Raises:
            AssertionError: If sequence doesn't match.

        Examples:
            >>> construct.verify_tool_sequence(["search", "summarize", "format"])
        """
        verify_sequence(
            calls=self._tool_calls,
            expected_sequence=expected_sequence,
            event_type="tool",
        )

    def verify_tools(
        self,
        count: int | None = None,
        min: int | None = None,
        max: int | None = None,
        target: str | tuple[str, ...] | None = None,
    ) -> None:
        """Verify tool calls with optional count/range and name filter.

        Args:
            count: Expected exact number of calls (mutually exclusive with min/max).
            min: Minimum number of calls (inclusive).
            max: Maximum number of calls (inclusive).
            target: Optional tool name filter. Can be a tuple of paths.

        Raises:
            AssertionError: If verification fails.
            ValueError: If count and min/max are both specified.

        Examples:
            >>> construct.verify_tools()  # at least one tool call
            >>> construct.verify_tools(count=3)  # exactly 3 tool calls
            >>> construct.verify_tools(min=2, max=4)  # between 2 and 4 calls
            >>> construct.verify_tools(target="search")  # at least one search call
            >>> construct.verify_tools(count=2, target="search")  # exactly 2
        """
        verify_call_count(
            calls=self._tool_calls,
            agent_runs=self._agent_runs,
            count=count,
            min=min,
            max=max,
            name_filter=target,
            agent_filter=None,
            event_type="tool",
        )


__all__ = ["ToolVerifications"]
