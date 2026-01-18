# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Verification API for construct testing.

Provides clean, expressive verifications for testing agent behavior:
- construct.verify_tool(fetch_data)  # at least once (1+)
- construct.verify_tools(count=1)  # exactly once
- construct.verify_tools(count=2)  # exactly twice
- construct.verify_llms(Provider.OPENAI, count=2)
- construct.verify_agent(risk_agent, threshold=0.8)
- construct.verify_llm(output_contains="success")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tenro._construct.verify.agent import AgentVerifications
from tenro._construct.verify.llm import LLMVerifications
from tenro._construct.verify.llm_scope import LLMScopeVerifications
from tenro._construct.verify.tool import ToolVerifications
from tenro.providers import Provider

if TYPE_CHECKING:
    from collections.abc import Callable

    from tenro._core.spans import AgentRun, LLMScope


class ConstructVerifications:
    """Verification API for testing agent behavior.

    Provides expressive verifications for tools, agents, and LLMs with support
    for count matching, argument checking, and sequence verification.
    """

    def __init__(
        self,
        agent_runs: list[AgentRun],
        llm_calls: list[Any] | None = None,
        tool_calls: list[Any] | None = None,
        llm_scopes: list[LLMScope] | None = None,
    ) -> None:
        """Initialize with hierarchical agent runs and optionally flat call lists.

        Args:
            agent_runs: List of agent runs (with populated .spans field).
            llm_calls: Optional pre-computed list of LLM calls (including orphans).
            tool_calls: Optional pre-computed list of tool calls (including orphans).
            llm_scopes: Optional list of LLMScope spans from @link_llm decorator.
        """
        if llm_calls is None:
            llm_calls = [llm for agent in agent_runs for llm in agent.get_llm_calls(recursive=True)]
        if tool_calls is None:
            tool_calls = [
                tool for agent in agent_runs for tool in agent.get_tool_calls(recursive=True)
            ]

        # Flatten agent runs to include nested agents for verification
        all_agents: list[AgentRun] = []
        for agent in agent_runs:
            all_agents.append(agent)
            all_agents.extend(agent.get_child_agents(recursive=True))

        self._tool_verifications = ToolVerifications(tool_calls, agent_runs)
        self._agent_verifications = AgentVerifications(all_agents)
        self._llm_verifications = LLMVerifications(llm_calls, agent_runs)
        self._llm_scope_verifications = LLMScopeVerifications(llm_scopes or [])

    def verify_tool(
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
        """Verify tool was called with optional argument and output matching.

        Args:
            target: Name or path of the tool. Can be a tuple of paths to match
                ANY of multiple entry method paths (for class targets).
            called_with: Dict of expected arguments.
            output: Expected response (dict=subset match, scalar=exact).
            output_contains: Expected substring in response.
            output_exact: Expected response (strict deep equality).
            call_index: Which call to check (0=first, -1=last, `None`=any).
            **kwargs: Expected keyword arguments.

        Raises:
            AssertionError: If verification fails.

        Examples:
            >>> construct.verify_tool(fetch_data)  # at least once
            >>> construct.verify_tool(get_weather, output={"temp": 72})  # subset
            >>> construct.verify_tool(search, output_contains="results")
        """
        self._tool_verifications.verify_tool(
            target,
            called_with,
            output=output,
            output_contains=output_contains,
            output_exact=output_exact,
            call_index=call_index,
            **kwargs,
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
        self._tool_verifications.verify_tool_never(target)

    def verify_tool_sequence(self, expected_sequence: list[str]) -> None:
        """Verify tools were called in a specific order.

        Args:
            expected_sequence: Expected sequence of tool names.

        Raises:
            AssertionError: If sequence doesn't match.

        Examples:
            >>> construct.verify_tool_sequence(["search", "summarize", "format"])
        """
        self._tool_verifications.verify_tool_sequence(expected_sequence)

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
        """
        self._tool_verifications.verify_tools(count=count, min=min, max=max, target=target)

    def verify_agent(
        self,
        target: str | tuple[str, ...],
        called_with: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Verify agent was called with optional argument matching.

        Args:
            target: Name of the agent. Can be a tuple of paths to match ANY
                of multiple entry method paths (for class targets).
            called_with: Dict of expected arguments.
            **kwargs: Expected keyword arguments.

        Raises:
            AssertionError: If verification fails.

        Examples:
            >>> construct.verify_agent(risk_agent)  # at least once
            >>> construct.verify_agent(risk_agent, threshold=0.8)  # with threshold
        """
        self._agent_verifications.verify_agent(target, called_with, **kwargs)

    def verify_agent_never(self, target: str | tuple[str, ...]) -> None:
        """Verify agent was never called.

        Args:
            target: Name of the agent. Can be a tuple of paths.

        Raises:
            AssertionError: If agent was called.

        Examples:
            >>> construct.verify_agent_never("DatabaseAgent")
        """
        self._agent_verifications.verify_agent_never(target)

    def verify_agent_sequence(self, expected_sequence: list[str]) -> None:
        """Verify agents were called in a specific order.

        Args:
            expected_sequence: Expected sequence of agent names.

        Raises:
            AssertionError: If sequence doesn't match.

        Examples:
            >>> construct.verify_agent_sequence(["Manager", "Researcher", "Writer"])
        """
        self._agent_verifications.verify_agent_sequence(expected_sequence)

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
            >>> construct.verify_agents(target="Manager")  # at least one Manager call
        """
        self._agent_verifications.verify_agents(count=count, min=min, max=max, target=target)

    def verify_llm(
        self,
        provider: Provider | str | None = None,
        *,
        target: str | None = None,
        count: int | None = None,
        output: Any = None,
        output_contains: str | None = None,
        output_exact: Any = None,
        where: str | None = None,
        call_index: int | None = None,
    ) -> None:
        """Verify LLM was called with optional output checking.

        Args:
            provider: Optional provider filter. Use Provider.OPENAI for built-ins.
            target: Optional target filter.
            count: Expected number of calls (`None` = at least once).
            output: Expected output (dict=subset match, scalar=exact).
            output_contains: Expected substring in output.
            output_exact: Expected output (strict deep equality).
            where: Selector (`None`=response, `"json"`, `"model"`).
            call_index: Which call to check (0=first, -1=last, `None`=any).

        Raises:
            AssertionError: If verification fails.

        Examples:
            >>> construct.verify_llm()  # at least once
            >>> construct.verify_llm(Provider.OPENAI)
            >>> construct.verify_llm(output_contains="success")
            >>> construct.verify_llm(where="json", output={"temp": 72})
        """
        self._llm_verifications.verify_llm(
            provider,
            target=target,
            count=count,
            output=output,
            output_contains=output_contains,
            output_exact=output_exact,
            where=where,
            call_index=call_index,
        )

    def verify_llm_never(
        self,
        provider: Provider | str | None = None,
        *,
        target: str | None = None,
    ) -> None:
        """Verify LLM was never called.

        Args:
            provider: Optional provider filter. Use Provider.OPENAI for built-ins.
            target: Optional target filter (specific function like
                `openai.chat.completions.create`).

        Raises:
            AssertionError: If LLM was called.

        Examples:
            >>> construct.verify_llm_never()
            >>> construct.verify_llm_never(Provider.ANTHROPIC)
            >>> construct.verify_llm_never(target="openai.chat.completions.create")
        """
        self._llm_verifications.verify_llm_never(provider, target=target)

    def verify_llms(
        self,
        provider: Provider | str | None = None,
        *,
        count: int | None = None,
        min: int | None = None,
        max: int | None = None,
        target: str | None = None,
    ) -> None:
        """Verify LLM calls with optional count/range and target/provider filter.

        Args:
            count: Expected exact number of calls (mutually exclusive with min/max).
            min: Minimum number of calls (inclusive).
            max: Maximum number of calls (inclusive).
            target: Optional target filter (specific function like
                `openai.chat.completions.create`).
            provider: Optional provider filter. Use Provider.OPENAI for built-ins.

        Raises:
            AssertionError: If verification fails.
            ValueError: If count and min/max are both specified.

        Examples:
            >>> construct.verify_llms()  # at least one LLM call
            >>> construct.verify_llms(count=2)  # exactly 2 LLM calls
            >>> construct.verify_llms(target="openai.chat.completions.create")
            >>> construct.verify_llms(Provider.OPENAI)
        """
        self._llm_verifications.verify_llms(
            count=count, min=min, max=max, target=target, provider=provider
        )

    def verify_llm_scope(
        self,
        target: Callable[..., Any] | None = None,
        *,
        times: int | None = None,
        scope_index: int | None = None,
        input: tuple[tuple[Any, ...], dict[str, Any]] | Any = None,
        input_contains: str | None = None,
        input_exact: Any = None,
        output: Any = None,
        output_contains: str | None = None,
        output_exact: Any = None,
    ) -> None:
        """Verify @link_llm function calls (LLMScope).

        Use this to verify what your @link_llm decorated function received and returned,
        as opposed to verify_llm() which checks HTTP-intercepted LLM calls.

        Args:
            target: Function reference to filter by (e.g., `extract_entities`).
            times: Expected number of calls. Mutually exclusive with scope_index.
            scope_index: Which call to check (0=first, -1=last). Exclusive with
                times.
            input: Expected input (smart match on (args, kwargs) pair).
            input_contains: Expected substring in input repr.
            input_exact: Expected input (strict equality).
            output: Expected output (dict=subset match, scalar=exact).
            output_contains: Expected substring in output repr.
            output_exact: Expected output (strict deep equality).

        Raises:
            AssertionError: If verification fails.
            TypeError: If mutually exclusive parameters are combined.

        Examples:
            >>> construct.verify_llm_scope(input_exact=(("hello",), {"lang": "en"}))
            >>> construct.verify_llm_scope(output={"status": "ok"})
            >>> construct.verify_llm_scope(target=extract_entities, times=1)
        """
        self._llm_scope_verifications.verify_llm_scope(
            target,
            times=times,
            scope_index=scope_index,
            input=input,
            input_contains=input_contains,
            input_exact=input_exact,
            output=output,
            output_contains=output_contains,
            output_exact=output_exact,
        )


__all__ = ["ConstructVerifications"]
