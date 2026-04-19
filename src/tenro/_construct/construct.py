# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Active construct for simulating agent tool calls and tracking lifecycle spans.

Provides lifecycle tracking, simulation helpers, provider-aware LLM simulation,
and verification APIs for tests.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from tenro._construct.http import HttpInterceptor
from tenro._construct.lifecycle import SpanAccessor, SpanLinker
from tenro._construct.simulate.orchestrator import SimulationOrchestrator
from tenro._construct.simulate.types import ResponsesInput
from tenro._construct.span_store import SpanCollector, SpanStore
from tenro._construct.verify.agent import AgentVerifications
from tenro._construct.verify.llm import LLMVerifications
from tenro._construct.verify.llm_scope import LLMScopeVerifications
from tenro._construct.verify.tool import ToolVerifications
from tenro._core.lifecycle_manager import LifecycleManager
from tenro._core.spans import AgentRun, LLMCall, ToolCall
from tenro.providers import _BUILTIN_PROVIDER_VALUES, Provider

logger = logging.getLogger(__name__)

_UNSET: Any = object()


class Construct:
    """Active construct for simulating tools and recording operation lifecycle.

    Records LLM calls and tool calls as mutable span objects that update in
    real-time, providing a simple API for testing.

    Features:
        - Simulate tool calls with smart defaults
        - Track LLM and tool calls as mutable span objects
        - Provider helpers for OpenAI, Anthropic, and Gemini
        - Expressive assertion API for readable tests

    See method docstrings (simulate_tool, simulate_agent, simulate_llm,
    verify_tool, verify_agent, verify_llm) for detailed examples and usage.
    """

    def __init__(
        self,
        allow_real_llm_calls: bool = False,
        fail_unused: bool = False,
    ) -> None:
        """Initialize construct for tracking and simulation.

        The construct wraps test code via context manager to track LLM and
        tool calls. Test code runs inside the `with construct:` block.

        Args:
            allow_real_llm_calls: Allow HTTP requests to LLM providers without
                simulations. When False (default), requests to known LLM
                domains without matching simulations raise `UnexpectedLLMCallError`,
                preventing accidental real API calls that cost money or leak
                credentials. Set to True for integration tests that intentionally
                call real APIs.
            fail_unused: If True, raise an error (instead of warning)
                when simulations are set up but never used. Controlled via
                --tenro-fail-unused pytest flag.

        Example:
            ```python
            # Default: blocks real LLM calls
            construct = Construct()

            # Allow real API calls (integration tests)
            construct = Construct(allow_real_llm_calls=True)
            ```
        """
        self._fail_unused = fail_unused
        self._span_store = SpanStore()
        self._trace_id: str | None = None

        effective_domains: list[str] | None = [] if allow_real_llm_calls else None

        self._handler = SpanCollector(self._span_store)
        self._lifecycle = LifecycleManager(handler=self._handler)
        self._http_interceptor = HttpInterceptor(
            on_call=self._handle_http_call,
            blocked_llm_domains=effective_domains,
        )

        self._orchestrator = SimulationOrchestrator(
            lifecycle=self._lifecycle,
            http_interceptor=self._http_interceptor,
        )
        self._linker = SpanLinker(lifecycle=self._lifecycle)
        self._span_accessor = SpanAccessor(span_store=self._span_store)

        self._provider_registry: dict[str, Provider] = {}  # normalized id → adapter
        self._default_provider_id: str | None = None  # always normalized string

    @property
    def agent_runs(self) -> list[AgentRun]:
        """Get all agent runs as a flat list.

        Returns:
            Flat list of all agent runs (includes nested agents).

        Examples:
            >>> agents = construct.agent_runs
            >>> assert len(agents) == 3  # Manager, Researcher, Writer
            >>> assert agents[0].name == "Manager"
            >>> assert agents[1].parent_agent_id == agents[0].id
        """
        return self._span_accessor.get_all_agent_runs()

    def _get_root_agent_runs(self) -> list[AgentRun]:
        """Get root agent runs with populated spans."""
        return self._span_accessor.get_root_agent_runs()

    @property
    def llm_calls(self) -> list[LLMCall]:
        """Get LLM calls across all agents and orphan calls.

        Returns:
            Flat list of LLM calls (includes orphan calls without agent parent).

        Examples:
            >>> llm_calls = construct.llm_calls
            >>> assert len(llm_calls) == 3
        """
        return self._span_accessor.get_llm_calls()

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Get tool calls across all agents and orphan calls.

        Returns:
            Flat list of tool calls (includes orphan calls without agent parent).

        Examples:
            >>> tool_calls = construct.tool_calls
            >>> assert len(tool_calls) == 2
        """
        return self._span_accessor.get_tool_calls()

    def _check_unused_llm_simulations(self) -> None:
        """Check for unused LLM simulations and raise errors immediately."""
        from tenro._construct.http.interceptor import get_supported_providers

        try:
            self._orchestrator.simulation_tracker.validate(
                llm_calls=self.llm_calls,
                supported_providers=list(get_supported_providers()),
                llm_scopes=self._span_accessor.get_llm_scopes(),
                strict=self._fail_unused,
            )
        finally:
            self._orchestrator.simulation_tracker.reset()

    def _check_unused_tool_simulations(self) -> None:
        """Check for unused tool simulations and raise errors immediately."""
        try:
            self._orchestrator.tool_tracker.validate("tool", strict=self._fail_unused)
        finally:
            self._orchestrator.tool_tracker.reset()

    def _check_unused_agent_simulations(self) -> None:
        """Check for unused agent simulations and raise errors immediately."""
        try:
            self._orchestrator.agent_tracker.validate("agent", strict=self._fail_unused)
        finally:
            self._orchestrator.agent_tracker.reset()

    @staticmethod
    def _normalize_provider_id(name: str) -> str:
        """Normalize provider ID to lowercase.

        Args:
            name: Raw provider name/ID.

        Returns:
            Normalized lowercase string.

        Raises:
            ValueError: If name is empty or whitespace-only.
        """
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Provider name cannot be empty or whitespace-only")
        return normalized

    def register_provider(self, name: str, *, adapter: Provider) -> None:
        """Register a custom provider with its HTTP adapter.

        Args:
            name: Custom provider identifier (e.g., "mistral").
                  Will be normalized to lowercase. Whitespace/empty rejected.
            adapter: Built-in provider whose HTTP format to use.

        Raises:
            ValueError: If name is empty, whitespace-only, or conflicts with built-in.
            ValueError: If name already registered with a different adapter.

        Examples:
            >>> construct.register_provider("mistral", adapter=Provider.OPENAI)
            >>> construct.simulate_llm("mistral", response="Hello!")
        """
        normalized = self._normalize_provider_id(name)
        if normalized in _BUILTIN_PROVIDER_VALUES:
            msg = (
                f"Cannot register '{normalized}': "
                f"conflicts with built-in Provider.{normalized.upper()}."
            )
            if name != normalized:
                msg += f" (from {name!r})"
            raise ValueError(msg)

        if normalized in self._provider_registry:
            existing = self._provider_registry[normalized]
            if existing != adapter:
                raise ValueError(
                    f"Provider '{normalized}' already registered with adapter "
                    f"Provider.{existing.name}; cannot re-register with Provider.{adapter.name}."
                )
            # Same adapter = idempotent, returns early
            return

        self._provider_registry[normalized] = adapter

    def set_default_provider(self, provider: Provider | str) -> None:
        """Set default provider for simulations.

        Args:
            provider: Built-in Provider enum or registered custom string.

        Raises:
            ValueError: If string is not registered.

        Examples:
            >>> construct.set_default_provider(Provider.OPENAI)
            >>> construct.register_provider("mistral", adapter=Provider.OPENAI)
            >>> construct.set_default_provider("mistral")
        """
        provider_id, _ = self._resolve_provider(provider=provider, target=None)
        self._default_provider_id = provider_id  # always store normalized string

    def _get_adapter(self, provider_id: str) -> Provider:
        """Get adapter for a validated provider ID.

        Args:
            provider_id: Normalized, validated provider ID (built-in or registered).

        Returns:
            Provider enum for HTTP layer.

        Raises:
            RuntimeError: If provider_id is unknown (invariant violation).
        """
        # provider_id is always normalized and validated before being stored
        if provider_id in _BUILTIN_PROVIDER_VALUES:
            return Provider(provider_id)
        try:
            return self._provider_registry[provider_id]
        except KeyError as e:
            raise RuntimeError(f"Invariant violated: unknown provider_id {provider_id!r}") from e

    def _validate_llm_params(
        self,
        response: Any,
        responses: Any,
        tool_calls: Any,
    ) -> None:
        """Validate simulate_llm parameters for common type mistakes.

        Raises:
            TypeError: If parameters have wrong types with helpful hints.
        """
        if response is not None and isinstance(response, list):
            raise TypeError(
                "Passed a list to 'response'. For multiple sequential responses "
                "use responses=[...]. For JSON list output use response='[...]' (stringified)."
            )

        if tool_calls is not None and not isinstance(tool_calls, list):
            raise TypeError(
                "Passed a single ToolCall to 'tool_calls'. Did you mean tool_calls=[...]?"
            )

    def _resolve_provider(
        self,
        *,
        provider: Provider | str | None,
        target: str | Callable[..., Any] | None,
    ) -> tuple[str, Provider]:
        """Resolve provider to (provider_id, adapter) tuple.

        Resolution order:
          1. If explicit provider= is provided → resolve it (or error)
          2. Else if target= implies a built-in adapter:
             - If default is set AND default's adapter == detected adapter →
               use default ID
             - Otherwise → use detected built-in provider ID
          3. Else if default provider is set → use it
          4. Else → error

        Args:
            provider: Explicit provider (enum or registered string), or None.
            target: Function/method for auto-detection, or None.

        Returns:
            Tuple of (provider_id, adapter) where:
            - provider_id: Normalized string identifier for filtering/logging
            - adapter: Provider enum for HTTP layer adapter selection

        Raises:
            ValueError: If provider string is unregistered or built-in used as string.
            ValueError: If no provider can be resolved.
        """
        # 1. Explicit provider
        if provider is not None:
            return self._resolve_explicit_provider(provider)

        # 2. Detect from target
        if target is not None:
            detected_adapter = self._detect_adapter_from_target(target)
            if detected_adapter is not None:
                if self._default_provider_id is not None:
                    default_adapter = self._get_adapter(self._default_provider_id)
                    if default_adapter == detected_adapter:
                        # Default matches detected → use default ID for grouping
                        return (self._default_provider_id, detected_adapter)
                # No default or adapters differ → use built-in provider ID
                return (detected_adapter.value, detected_adapter)

        # 3. Default provider
        if self._default_provider_id is not None:
            adapter = self._get_adapter(self._default_provider_id)
            return (self._default_provider_id, adapter)

        # 4. Error
        raise ValueError("No provider set. Pass provider=... or call set_default_provider(...)")

    def _resolve_explicit_provider(self, provider: Provider | str) -> tuple[str, Provider]:
        """Resolve an explicitly provided provider value.

        Args:
            provider: Provider enum or string.

        Returns:
            Tuple of (provider_id, adapter).

        Raises:
            ValueError: If string is unregistered or built-in used as string.
            ValueError: If string looks like a target (contains ".").
        """
        if isinstance(provider, str) and "." in provider:
            raise ValueError(
                f"It looks like you passed a target string positionally. "
                f"Use target='{provider}' instead."
            )
        if isinstance(provider, Provider):
            return (provider.value, provider)

        normalized = self._normalize_provider_id(provider)
        if normalized in _BUILTIN_PROVIDER_VALUES:
            raise ValueError(
                f"Use Provider.{normalized.upper()} (enum) for built-in providers, "
                f"not the string '{provider}'"
            )
        if normalized in self._provider_registry:
            return (normalized, self._provider_registry[normalized])
        return self._resolve_registered_provider(normalized, provider)

    def _resolve_registered_provider(
        self, normalized: str, original: Provider | str
    ) -> tuple[str, Provider]:
        """Look up a custom/registry provider by normalized ID."""
        from tenro._construct.http.registry import ProviderRegistry
        from tenro._construct.http.registry.exceptions import UnsupportedProviderError

        try:
            config = ProviderRegistry.get_provider(normalized)
            family_to_adapter = {
                "openai_compatible": Provider.OPENAI,
                "anthropic_compatible": Provider.ANTHROPIC,
                "gemini_compatible": Provider.GEMINI,
            }
            adapter = family_to_adapter.get(config.compatibility_family)
            if adapter:
                return (normalized, adapter)
        except UnsupportedProviderError:
            pass

        msg = f"Unknown provider '{normalized}'."
        if original != normalized:
            msg += f" (from {original!r})"
        msg += " Register it first with tenro.register_provider()"
        raise ValueError(msg)

    def _detect_adapter_from_target(self, target: str | Callable[..., Any]) -> Provider | None:
        """Detect built-in provider adapter from target.

        Args:
            target: Target path or callable.

        Returns:
            Provider enum if detected, None otherwise.
        """
        from tenro._construct.http.builders import ProviderSchemaFactory

        if callable(target) and not isinstance(target, str):
            qualname = getattr(target, "__qualname__", "")
            module = getattr(target, "__module__", "")
            target_path = f"{module}.{qualname}" if module and qualname else str(target)
        else:
            target_path = str(target)

        try:
            detected = ProviderSchemaFactory.detect_provider(target_path)
            if detected in _BUILTIN_PROVIDER_VALUES:
                return Provider(detected)
        except ValueError:
            pass
        return None

    def _normalize_provider_for_filter(self, provider: Provider | str | None) -> str | None:
        """Normalize provider for verification filtering.

        Verification mirrors simulation: built-in strings require enum,
        custom strings must be registered. Empty/whitespace = no filter.

        Args:
            provider: Provider enum, string, or None.

        Returns:
            Normalized provider string, or None if no filter.

        Raises:
            ValueError: If string is a built-in (use enum) or unregistered.
        """
        if provider is None:
            return None
        if isinstance(provider, Provider):
            return provider.value

        normalized = provider.strip().lower()
        if not normalized:
            return None

        if normalized in _BUILTIN_PROVIDER_VALUES:
            raise ValueError(
                f"Use Provider.{normalized.upper()} (enum) for built-in providers, "
                f"not the string '{provider}'"
            )

        if normalized in self._provider_registry:
            return normalized

        from tenro._construct.http.registry import ProviderRegistry
        from tenro._construct.http.registry.exceptions import UnsupportedProviderError

        try:
            ProviderRegistry.get_provider(normalized)
            return normalized
        except UnsupportedProviderError:
            pass

        msg = f"Unknown provider '{normalized}'."
        if provider != normalized:
            msg += f" (from {provider!r})"
        msg += " Register it first with tenro.register_provider()"
        raise ValueError(msg)

    def link_agent(
        self, name: str, input_data: Any = None, **kwargs: Any
    ) -> AbstractContextManager[AgentRun]:
        """Link an agent execution with automatic lifecycle management.

        Creates an AgentRun span with automatic parent tracking, latency
        calculation, and stack-safe cleanup.

        Args:
            name: Agent name.
            input_data: Input data for the agent.
            **kwargs: Additional keyword arguments passed to the agent.

        Returns:
            Context manager yielding mutable AgentRun span.

        Examples:
            >>> with construct.link_agent("Manager", input_data="task") as agent:
            ...     # Nested operations automatically get Manager as parent
            ...     agent.output_data = "completed"
            >>> with construct.link_agent("RiskAgent", threshold=0.8) as agent:
            ...     # Agent called with threshold parameter
            ...     agent.output_data = "low risk"
        """
        return self._linker.link_agent(name, input_data, **kwargs)

    def link_tool(
        self, tool_name: str, *args: Any, **kwargs: Any
    ) -> AbstractContextManager[ToolCall]:
        """Link a tool call with automatic lifecycle management.

        Creates a ToolCall span with automatic parent tracking, latency
        calculation, and stack-safe cleanup.

        Args:
            tool_name: Name of the tool being called.
            *args: Positional arguments passed to the tool.
            **kwargs: Keyword arguments passed to the tool.

        Returns:
            Context manager yielding mutable ToolCall span.

        Examples:
            >>> with construct.link_tool("search") as tool:
            ...     tool.result = ["doc1", "doc2"]
            >>> with construct.link_tool("search", "query", limit=10) as tool:
            ...     tool.result = ["doc1", "doc2"]
            >>> with construct.link_tool("api_call", "POST", timeout=30) as tool:
            ...     tool.result = {"status": 200}
        """
        return self._linker.link_tool(tool_name, *args, **kwargs)

    def simulate_tool(
        self,
        target: str | Callable[..., Any],
        result: Any = None,
        results: list[Any] | None = None,
        side_effect: Callable[..., Any] | None = None,
        optional: bool = False,
    ) -> None:
        """Simulate an agent tool with lifecycle recording.

        Automatically creates ToolCall spans to record execution.

        Args:
            target: Dotted-path string to a function/method (e.g.,
                "myapp.tools.Tool.run") or a callable function/method object.
            result: Single static value for tool to produce (most common case).
            results: List of values for sequential calls. Can include Exception
                objects which will be raised when reached.
            side_effect: Callable for dynamic behavior. Receives tool arguments.
            optional: If True, won't raise if the simulation is never triggered.

        Raises:
            ValueError: If multiple result parameters are provided.

        Examples:
            >>> # Single result (most common)
            >>> construct.simulate_tool("myapp.tools.search", result="doc1")
            >>>
            >>> # Sequential results with exceptions
            >>> construct.simulate_tool(
            ...     "myapp.tools.api_call",
            ...     results=[
            ...         {"status": "ok"},
            ...         TimeoutError("Connection lost"),
            ...         {"status": "ok"},
            ...     ],
            ... )
            >>>
            >>> # Dynamic behavior
            >>> def weather_logic(city: str):
            ...     return {"temp": 72 if city == "SF" else 65}
            >>> construct.simulate_tool(
            ...     "myapp.tools.get_weather", side_effect=weather_logic
            ... )
            >>>
            >>> # Function object (refactor-safe)
            >>> from myapp.tools import search
            >>> construct.simulate_tool(search, result="doc1")
        """
        self._orchestrator.simulate_tool(target, result, results, side_effect, optional)

    def simulate_agent(
        self,
        target: str | Callable[..., Any],
        result: Any = None,
        results: list[Any] | None = None,
        side_effect: Callable[..., Any] | None = None,
        optional: bool = False,
    ) -> None:
        """Simulate an agent with lifecycle recording.

        Agents are high-level workflows that orchestrate tools and LLMs.
        This method enables testing agent behavior in isolation.

        Args:
            target: Dotted-path string to a function/method (e.g.,
                "myapp.agents.Planner.run") or a callable function/method object.
            result: Single static value for agent to produce (most common case).
            results: List of values for sequential calls. Can include Exception
                objects which will be raised when reached.
            side_effect: Callable for dynamic behavior. Receives agent arguments.
            optional: If True, won't raise if the simulation is never triggered.

        Raises:
            ValueError: If multiple result parameters are provided.

        Examples:
            >>> # Single result (most common)
            >>> construct.simulate_agent(
            ...     "myapp.agents.planner", result={"plan": "step1"}
            ... )
            >>>
            >>> # Sequential results with exceptions
            >>> construct.simulate_agent(
            ...     "myapp.agents.researcher",
            ...     results=[
            ...         {"findings": "data1"},
            ...         TimeoutError("Research timeout"),
            ...         {"findings": "retry_data"},
            ...     ],
            ... )
            >>>
            >>> # Dynamic behavior
            >>> def agent_logic(query: str):
            ...     return {"result": f"processed_{query}"}
            >>> construct.simulate_agent(
            ...     "myapp.agents.processor", side_effect=agent_logic
            ... )
            >>>
            >>> # Function object (refactor-safe)
            >>> from myapp.agents import planner
            >>> construct.simulate_agent(planner, result={"plan": "step1"})
        """
        self._orchestrator.simulate_agent(target, result, results, side_effect, optional)

    def __enter__(self) -> Construct:
        """Activate simulations and start HTTP interception.

        Registers this construct as the active construct for decorator access.
        After activation, subsequent simulate_* calls apply patches immediately.
        """
        from tenro.linking.decorators import _set_active_construct

        _set_active_construct(self)
        self._orchestrator.activate()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _exc_traceback: Any,
    ) -> None:
        """Restore original functions/methods and cleanup state.

        Harness errors (e.g., unused simulations) are raised immediately.

        Note:
            Never suppresses incoming exceptions from test body.
        """
        from tenro.linking.decorators import _set_active_construct

        _set_active_construct(None)
        try:
            self._check_unused_llm_simulations()
            self._check_unused_tool_simulations()
            self._check_unused_agent_simulations()
        finally:
            self._orchestrator.deactivate()

    def _tool_verifier(self) -> ToolVerifications:
        """Create tool verifier from current spans."""
        return ToolVerifications(self.tool_calls, self._get_root_agent_runs())

    def _agent_verifier(self) -> AgentVerifications:
        """Create agent verifier from current spans."""
        all_agents: list[AgentRun] = []
        for agent in self._get_root_agent_runs():
            all_agents.append(agent)
            all_agents.extend(agent.get_child_agents(recursive=True))
        return AgentVerifications(all_agents)

    def _llm_verifier(self) -> LLMVerifications:
        """Create LLM verifier from current spans."""
        return LLMVerifications(self.llm_calls, self._get_root_agent_runs())

    def _llm_scope_verifier(self) -> LLMScopeVerifications:
        """Create LLM scope verifier from current spans."""
        return LLMScopeVerifications(self._span_accessor.get_llm_scopes())

    def verify_tool(
        self,
        target: str | Callable[..., Any],
        *,
        result: Any = _UNSET,
        where: Callable[[Any], bool] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Verify tool was called with optional result matching.

        Args:
            target: Name, path, or function object of the tool.
            result: Expected result value (exact equality check). Use to verify
                the tool returned a specific value including None.
            where: Predicate filter for span selection.
            **kwargs: Additional arguments for matching tool arguments.

        Returns:
            The matching ToolCallSpan.

        Raises:
            AssertionError: If verification fails.

        Examples:
            >>> construct.verify_tool(fetch_data)  # at least once
            >>> construct.verify_tool(fetch_data, query="test")  # with argument match
            >>> construct.verify_tool(get_weather, result={"temp": 72})
        """
        from tenro._construct.simulate.target_resolution import (
            resolve_all_target_paths,
        )

        resolved_target = resolve_all_target_paths(target, is_tool=True)
        if result is not _UNSET:
            kwargs["output_exact"] = result
        self._tool_verifier().verify_tool(resolved_target, **kwargs)
        filtered = [t for t in self.tool_calls if t.target_path in resolved_target]
        if where:
            filtered = [t for t in filtered if where(t)]
        return filtered[0] if filtered else None

    def verify_tool_never(self, target: str | Callable[..., Any]) -> None:
        """Verify tool was never called.

        Args:
            target: Name, path, or function object of the tool.

        Raises:
            AssertionError: If tool was called.

        Examples:
            >>> construct.verify_tool_never("dangerous_operation")
            >>> construct.verify_tool_never(dangerous_operation)
        """
        from tenro._construct.simulate.target_resolution import (
            resolve_all_target_paths,
        )

        resolved_target = resolve_all_target_paths(target, is_tool=True)
        self._tool_verifier().verify_tool_never(resolved_target)

    def verify_tool_sequence(self, expected_sequence: list[str]) -> None:
        """Verify tools were called in a specific order.

        Args:
            expected_sequence: Expected sequence of tool names.

        Raises:
            AssertionError: If sequence doesn't match.

        Examples:
            >>> construct.verify_tool_sequence(["search", "summarize", "format"])
        """
        self._tool_verifier().verify_tool_sequence(expected_sequence)

    def verify_tools(
        self,
        count: int | None = None,
        min: int | None = None,
        max: int | None = None,
        target: str | None = None,
    ) -> None:
        """Verify tool calls with optional count/range and name filter.

        Args:
            count: Expected exact number of calls (mutually exclusive with min/max).
            min: Minimum number of calls (inclusive).
            max: Maximum number of calls (inclusive).
            target: Optional tool name filter.

        Raises:
            AssertionError: If verification fails.
            ValueError: If count and min/max are both specified.

        Examples:
            >>> construct.verify_tools()  # at least one tool call
            >>> construct.verify_tools(count=3)  # exactly 3 tool calls
            >>> construct.verify_tools(min=2, max=4)  # between 2 and 4 calls
        """
        self._tool_verifier().verify_tools(count=count, min=min, max=max, target=target)

    def verify_agent(
        self,
        target: str | Callable[..., Any],
        *,
        result: Any = _UNSET,
        where: Callable[[Any], bool] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Verify agent was called with optional result matching.

        Args:
            target: Name, path, or function object of the agent.
            result: Expected result value (exact equality check). Use to verify
                the agent returned a specific value including None.
            where: Predicate filter for span selection.
            **kwargs: Additional arguments for matching tool arguments.

        Returns:
            The matching AgentRunSpan.

        Raises:
            AssertionError: If verification fails.

        Examples:
            >>> construct.verify_agent(risk_agent)  # at least once
            >>> construct.verify_agent(risk_agent, result={"score": 0.8})
        """
        from tenro._construct.simulate.target_resolution import (
            resolve_all_target_paths,
        )

        resolved_target = resolve_all_target_paths(target)
        if result is not _UNSET:
            kwargs["output_exact"] = result
        self._agent_verifier().verify_agent(resolved_target, **kwargs)
        filtered = [a for a in self.agent_runs if a.target_path in resolved_target]
        if where:
            filtered = [a for a in filtered if where(a)]
        return filtered[0] if filtered else None

    def verify_agent_never(self, target: str | Callable[..., Any]) -> None:
        """Verify agent was never called.

        Args:
            target: Name, path, or function object of the agent.

        Raises:
            AssertionError: If agent was called.

        Examples:
            >>> construct.verify_agent_never("FallbackAgent")
            >>> construct.verify_agent_never(fallback_agent)
        """
        from tenro._construct.simulate.target_resolution import (
            resolve_all_target_paths,
        )

        resolved_target = resolve_all_target_paths(target)
        self._agent_verifier().verify_agent_never(resolved_target)

    def verify_agent_sequence(self, expected_sequence: list[str]) -> None:
        """Verify agents were called in a specific order.

        Args:
            expected_sequence: Expected sequence of agent names.

        Raises:
            AssertionError: If sequence doesn't match.

        Examples:
            >>> construct.verify_agent_sequence(["Planner", "Executor", "Reviewer"])
        """
        self._agent_verifier().verify_agent_sequence(expected_sequence)

    def verify_agents(
        self,
        count: int | None = None,
        min: int | None = None,
        max: int | None = None,
        target: str | Callable[..., Any] | None = None,
    ) -> None:
        """Verify agent calls with optional count/range and name filter.

        Args:
            count: Expected exact number of calls (mutually exclusive with min/max).
            min: Minimum number of calls (inclusive).
            max: Maximum number of calls (inclusive).
            target: Optional agent name, path, or callable reference to filter.

        Raises:
            AssertionError: If verification fails.
            ValueError: If count and min/max are both specified.

        Examples:
            >>> construct.verify_agents()  # at least one agent call
            >>> construct.verify_agents(count=2)  # exactly 2 agent calls
        """
        from tenro._construct.simulate.target_resolution import resolve_all_target_paths

        resolved_target = resolve_all_target_paths(target) if target else None
        self._agent_verifier().verify_agents(count=count, min=min, max=max, target=resolved_target)

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
            provider: Optional provider filter. Use Provider.OPENAI for built-ins
                or registered string ID for custom providers.
            target: Optional target filter (e.g., "openai.chat.completions.create").
            count: Expected number of calls (None = at least once).
            output: Expected output (dict=subset match, scalar=exact).
            output_contains: Expected substring in response text.
            output_exact: Expected output (strict deep equality).
            where: Output location to check (None=response, "json"=parsed JSON).
            call_index: Which call to check (0=first, -1=last, None=any).

        Raises:
            AssertionError: If verification fails.

        Examples:
            >>> construct.verify_llm()  # at least one LLM call
            >>> construct.verify_llm(Provider.OPENAI)
            >>> construct.verify_llm(output_contains="weather")
        """
        provider_str = self._normalize_provider_for_filter(provider)
        self._llm_verifier().verify_llm(
            provider_str,
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
            provider: Optional provider filter. Use Provider.OPENAI for built-ins
                or registered string ID for custom providers.
            target: Optional target filter (e.g., "openai.chat.completions.create").

        Raises:
            AssertionError: If LLM was called.

        Examples:
            >>> construct.verify_llm_never()  # no LLM calls at all
            >>> construct.verify_llm_never(Provider.ANTHROPIC)
        """
        provider_str = self._normalize_provider_for_filter(provider)
        self._llm_verifier().verify_llm_never(provider_str, target=target)

    def verify_llms(
        self,
        provider: Provider | str | None = None,
        *,
        count: int | None = None,
        min: int | None = None,
        max: int | None = None,
        target: str | None = None,
    ) -> None:
        """Verify LLM calls with optional count/range and filters.

        Args:
            provider: Optional provider filter. Use Provider.OPENAI for built-ins
                or registered string ID for custom providers.
            count: Expected exact number of calls (mutually exclusive with min/max).
            min: Minimum number of calls (inclusive).
            max: Maximum number of calls (inclusive).
            target: Optional target path filter.

        Raises:
            AssertionError: If verification fails.
            ValueError: If count and min/max are both specified.

        Examples:
            >>> construct.verify_llms()  # at least one LLM call
            >>> construct.verify_llms(count=3)  # exactly 3 LLM calls
            >>> construct.verify_llms(Provider.ANTHROPIC)
        """
        provider_str = self._normalize_provider_for_filter(provider)
        self._llm_verifier().verify_llms(
            count=count, min=min, max=max, target=target, provider=provider_str
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
            times: Expected number of calls. Exclusive with scope_index.
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
            >>> construct.verify_llm_scope(
            ...     input_exact=(("hello",), {"lang": "en"})
            ... )
            >>> construct.verify_llm_scope(output={"status": "ok"})
            >>> construct.verify_llm_scope(target=extract_entities, times=1)
        """
        self._llm_scope_verifier().verify_llm_scope(
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

    def simulate_llm(
        self,
        provider: Provider | str | None = None,
        *,
        target: str | Callable[..., Any] | None = None,
        response: str | None = None,
        responses: ResponsesInput = None,
        model: str | None = None,
        tool_calls: list[Any] | None = None,
        use_http: bool | None = None,
        optional: bool = False,
        **response_kwargs: Any,
    ) -> None:
        """Simulate LLM calls with smart provider detection and lifecycle recording.

        This method keeps the API simple while preserving provider-accurate
        responses when possible:
        - Accepts plain text responses
        - Auto-detects known provider SDK targets
        - Uses HTTP interception for known providers (default without custom target)
        - Falls back to dispatch mode for @link_llm or registered targets
        - Creates LLMCall spans to record execution

        HTTP interception ensures the provider SDK parses JSON and yields
        real SDK types (e.g., `TextBlock`, `ChatCompletion`).

        Supports text generation APIs only. Supported targets include:
        - OpenAI Chat Completions (`openai.chat.completions.create`)
        - Anthropic Messages (`anthropic.resources.messages.Messages.create`)
        - Gemini GenerateContent (`google.genai.models.Models.generate_content`)

        For other provider APIs (embeddings, audio, images, assistants),
        register a custom provider schema or use a custom target.

        Args:
            provider: LLM provider. Use Provider.OPENAI, Provider.ANTHROPIC, or
                Provider.GEMINI for built-in providers. For custom providers,
                register first with register_provider() then pass the string ID.
                Auto-detected from target if not provided.
            target: Optional dotted-path string to a function/method (e.g.,
                "openai.chat.completions.create"). If not provided, uses the
                default target for the specified provider.
            response: Single string response (most common case). Deprecated;
                prefer responses= for all new code.
            responses: Responses for sequential calls. Accepts:
                - str: Simple text response
                - ToolCall: Single tool call (e.g., ToolCall(search, q="AI"))
                - list[ToolCall]: Multiple tool calls in one turn
                - list[str | ToolCall]: Interleaved text and tool calls
                - LLMResponse: Explicit block structure for complex responses
                - Exception: Will be raised when this turn is reached
                - dict: Legacy {text?, tool_calls?} format (deprecated)
            model: Model identifier (e.g., "gpt-4", "claude-3-opus").
                Overrides provider default model name.
            tool_calls: Tool calls the LLM should emit. Accepts:
                - ToolCall objects: ToolCall("get_weather", city="Paris")
                - Dicts: {"name": "tool", "arguments": {...}}
                - Strings (name only): "get_weather"
                Use ToolCall(my_func, **kwargs) to reference tools by function.
            use_http: Force HTTP interception (True) or dispatch mode (False).
                Defaults to True for known providers. When False, target must
                be decorated with @link_llm or registered via tenro.simulate.register().
            optional: If True, this simulation won't cause UnusedSimulationError
                if unused. Use for branch coverage or optional paths. Defaults
                to False.
            **response_kwargs: Provider-specific options:
                - token_usage: Dict with token counts (e.g., {"total_tokens": 50})
                - finish_reason: Override finish reason ("stop", "length", "tool_calls")
                - stop_reason: Anthropic-specific stop reason
                - safety_ratings: Gemini-specific safety ratings
                - (Other provider-specific parameters)

        Raises:
            ValueError: If both `response` and `responses` are provided.
            ValueError: If provider string is not registered or uses built-in name.

        Examples:
            >>> # Simple text response
            >>> construct.simulate_llm(Provider.ANTHROPIC, responses="Hello!")
            >>>
            >>> # Single tool call (shorthand)
            >>> construct.simulate_llm(
            ...     Provider.OPENAI,
            ...     responses=ToolCall(search, query="AI"),
            ... )
            >>>
            >>> # Multiple tool calls in one turn
            >>> construct.simulate_llm(
            ...     Provider.OPENAI,
            ...     responses=[[ToolCall(search, q="A"), ToolCall(fetch, id=1)]],
            ... )
            >>>
            >>> # Interleaved text and tool calls
            >>> construct.simulate_llm(
            ...     Provider.ANTHROPIC,
            ...     responses=[["Let me search", ToolCall(search, q="AI"), "Done!"]],
            ... )
            >>>
            >>> # Multi-turn conversation
            >>> construct.simulate_llm(
            ...     Provider.ANTHROPIC,
            ...     responses=["Turn 1", "Turn 2", "Turn 3"],
            ... )
            >>>
            >>> # Custom provider (register first)
            >>> construct.register_provider("mistral", adapter=Provider.OPENAI)
            >>> construct.simulate_llm("mistral", responses="Hello!")
            >>>
            >>> # Optional simulation for branch coverage
            >>> construct.simulate_llm(
            ...     Provider.OPENAI,
            ...     responses="Fallback response",
            ...     optional=True,
            ... )
        """
        from tenro._construct.simulate.orchestrator import (
            _is_linked_type,
            _is_registered_target,
            _resolve_string_target,
        )

        self._validate_llm_params(response, responses, tool_calls)

        resolved_target = _resolve_string_target(target) if isinstance(target, str) else target
        is_simulatable_target = (
            resolved_target is not None
            and callable(resolved_target)
            and (_is_linked_type(resolved_target, "llm") or _is_registered_target(resolved_target))
        )

        if use_http is False and resolved_target is not None and callable(resolved_target):
            self._assert_simulatable_target(resolved_target, is_simulatable_target, target)

        if use_http is False and is_simulatable_target:
            provider_id, adapter_value = "custom", "openai"
        else:
            provider_id, adapter = self._resolve_provider(provider=provider, target=target)
            adapter_value = adapter.value

        self._orchestrator.simulate_llm(
            target=target,
            provider=provider_id,
            adapter=adapter_value,
            response=response,
            responses=responses,
            model=model,
            tool_calls=tool_calls,
            use_http=use_http,
            optional=optional,
            **response_kwargs,
        )

    def _assert_simulatable_target(
        self,
        resolved: object,
        is_simulatable: bool,
        original: str | Callable[..., Any] | None,
    ) -> None:
        """Raise if use_http=False target is not linked or registered."""
        if is_simulatable:
            return
        from tenro._construct.simulate.orchestrator import _detect_target_type
        from tenro.errors.simulation import SimulationDiagnostic, TenroSimulationSetupError

        target_name = getattr(resolved, "__qualname__", repr(original))
        diagnostic = SimulationDiagnostic(
            target_path=target_name,
            target_type=_detect_target_type(resolved),
            is_linked=False,
            failure_reason=(
                "Target is not linked or registered; capture-safe interception unavailable."
            ),
            recommended_fix=(
                "Decorate with @link_llm, use tenro.register(), or use HTTP mode (use_http=True)."
            ),
        )
        raise TenroSimulationSetupError(
            f"Cannot simulate non-linked LLM target '{target_name}'.\n\n"
            f"Reason: {diagnostic.failure_reason}\n"
            f"Fix: {diagnostic.recommended_fix}",
            diagnostic=diagnostic,
        )

    def _handle_http_call(
        self,
        provider: str,
        messages: list[dict[str, Any]],
        model: str | None,
        response_text: str,
        agent: str | None,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        """Callback for HTTP interception - delegates to orchestrator."""
        self._orchestrator.handle_http_call(
            provider, messages, model, response_text, agent, tool_calls
        )
