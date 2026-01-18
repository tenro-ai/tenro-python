# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Agent verification methods for construct testing.

Provides 4 core verification methods for agent runs:
- verify_agent() - flexible verification with argument matching
- verify_agent_never() - explicit "never called" check
- verify_agent_sequence() - order verification
- verify_agents() - aggregate/range queries
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
    from tenro._core.spans import AgentRun


class AgentVerifications:
    """Agent verification methods."""

    def __init__(self, agent_runs: list[AgentRun]) -> None:
        """Initialize with agent runs.

        Args:
            agent_runs: List of agent run spans.
        """
        self._agent_runs = agent_runs

    def verify_agent(
        self,
        target: str | tuple[str, ...],
        called_with: dict[str, Any] | None = None,
        *,
        output: Any = None,
        output_contains: str | None = None,
        output_exact: Any = None,
        call_index: int | None = 0,
        **kwargs: Any,
    ) -> None:
        """Verify agent was called with optional argument and output matching.

        Args:
            target: Name of the agent. Can be a tuple of paths to match ANY
                of multiple entry method paths (for class targets).
            called_with: Dict of expected arguments.
            output: Expected output (dict=subset match, scalar=exact).
            output_contains: Expected substring in output (strings only).
            output_exact: Expected output (strict deep equality).
            call_index: Which call to check (0=first, -1=last, None=any).
            **kwargs: Expected keyword arguments.

        Raises:
            AssertionError: If verification fails.

        Examples:
            >>> construct.verify_agent(risk_agent)  # at least once
            >>> construct.verify_agent(risk_agent, threshold=0.8)  # with arg
            >>> construct.verify_agent(bot_agent, output={"status": "ok"})
            >>> construct.verify_agent(bot_agent, output_contains="success")
            >>> construct.verify_agent(bot_agent, output_exact={"a": 1})
        """
        targets = (target,) if isinstance(target, str) else target

        matching_runs = [r for r in self._agent_runs if r.target_path in targets]

        # Output verification (takes precedence)
        if output is not None:
            verify_output(matching_runs, "output_data", output, "agent", call_index=call_index)
            return
        if output_contains is not None:
            verify_output_contains(
                matching_runs,
                "output_data",
                output_contains,
                "agent",
                call_index=call_index,
            )
            return
        if output_exact is not None:
            verify_output_exact(
                matching_runs,
                "output_data",
                output_exact,
                "agent",
                call_index=call_index,
            )
            return

        # Input/count verification
        if not called_with and not kwargs:
            verify_call_count(
                calls=self._agent_runs,
                agent_runs=self._agent_runs,
                count=None,
                min=1,
                max=None,
                name_filter=target,
                agent_filter=None,
                event_type="agent",
            )
        else:
            verify_with_arguments(
                calls=self._agent_runs,
                agent_runs=self._agent_runs,
                name=target,
                called_with=called_with,
                agent_filter=None,
                event_type="agent",
                kwargs=kwargs,
            )

    def verify_agent_never(self, target: str | tuple[str, ...]) -> None:
        """Verify agent was never called.

        Args:
            target: Name of the agent. Can be a tuple of paths.

        Raises:
            AssertionError: If agent was called.

        Examples:
            >>> construct.verify_agent_never("DeprecatedAgent")
        """
        verify_call_count(
            calls=self._agent_runs,
            agent_runs=self._agent_runs,
            count=0,
            min=None,
            max=None,
            name_filter=target,
            agent_filter=None,
            event_type="agent",
        )

    def verify_agent_sequence(self, expected_sequence: list[str]) -> None:
        """Verify agents were called in a specific order.

        Args:
            expected_sequence: Expected sequence of agent names.

        Raises:
            AssertionError: If sequence doesn't match.

        Examples:
            >>> construct.verify_agent_sequence(["Manager", "Researcher", "Writer"])
        """
        verify_sequence(
            calls=self._agent_runs,
            expected_sequence=expected_sequence,
            event_type="agent",
        )

    def verify_agents(
        self,
        count: int | None = None,
        min: int | None = None,
        max: int | None = None,
        target: str | tuple[str, ...] | None = None,
    ) -> None:
        """Verify agent calls with optional count/range and name filter.

        Args:
            count: Expected exact number of calls (mutually exclusive with min/max).
            min: Minimum number of calls (inclusive).
            max: Maximum number of calls (inclusive).
            target: Optional agent name filter. Can be a tuple of paths.

        Raises:
            AssertionError: If verification fails.
            ValueError: If count and min/max are both specified.

        Examples:
            >>> construct.verify_agents()  # at least one agent call
            >>> construct.verify_agents(count=3)  # exactly 3 agent calls
            >>> construct.verify_agents(min=2, max=4)  # between 2 and 4 calls
            >>> construct.verify_agents(target="Manager")  # at least one Manager call
            >>> construct.verify_agents(count=2, target="Researcher")  # exactly 2 calls
        """
        verify_call_count(
            calls=self._agent_runs,
            agent_runs=self._agent_runs,
            count=count,
            min=min,
            max=max,
            name_filter=target,
            agent_filter=None,
            event_type="agent",
        )


__all__ = ["AgentVerifications"]
