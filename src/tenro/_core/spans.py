# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Span models for tracking LLM calls, tool calls, and agent runs.

These objects track lifecycle during test execution, starting in
"running" state and updating in-place as operations complete.

Span types:
- LLMCall: Tracks an LLM API call
- ToolCall: Tracks a tool invocation
- AgentRun: Tracks an agent execution
- LLMScope: Transparent annotation span for @link_llm decorator
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from tenro._core.model_base import BaseModel


class BaseSpan(BaseModel):
    """Base class for all span types (LLMCall, ToolCall, AgentRun).

    Defines shared fields for timed operations in a trace. Spans share the
    same lifecycle: running -> completed/error.

    Attributes:
        id: Unique span identifier.
        trace_id: Trace identifier for the overall workflow.
        start_time: Unix timestamp when the span started.
        parent_span_id: Immediate parent span ID (Agent, LLM, or Tool).
        agent_id: Closest agent ancestor span ID, if any.
        status: Lifecycle status (`running`, `completed`, or `error`).
        latency_ms: Span duration in milliseconds.
        error: Error message if the span failed.
        metadata: Additional metadata for the span.
    """

    id: str
    trace_id: str
    start_time: float

    parent_span_id: str | None = None
    agent_id: str | None = None
    status: Literal["running", "completed", "error"] = "running"
    latency_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMCall(BaseSpan):
    """Mutable object representing an LLM API call lifecycle.

    Created when LLM call starts, updated in-place as it progresses.
    Response, latency, and status are accessible at any time.

    Attributes:
        provider: LLM provider name (e.g., `openai`, `anthropic`).
        messages: List of message dicts sent to the provider.
        response: Model response text, when available.
        model: Model identifier used for the call.
        token_usage: Token usage metadata when available.
        tool_calls: Tool calls emitted by the LLM in its response, if any.
            Each dict contains tool name, id, and arguments.
        caller_signature: Caller function signature for error messages.
        caller_location: Caller file location for error messages.
        agent_name: Name of the @link_agent-decorated agent that made this call,
            or None if called outside of an agent context.
        llm_scope_id: ID of enclosing LLMScope from @link_llm decorator, if any.
    """

    provider: str
    messages: list[dict[str, Any]]

    response: str | None = None
    model: str | None = None
    token_usage: dict[str, int] | None = None
    tool_calls: list[dict[str, Any]] | None = None

    caller_signature: str | None = None
    caller_location: str | None = None
    agent_name: str | None = None
    llm_scope_id: str | None = None
    target_path: str | None = None
    simulated: bool = False


class LLMScope(BaseSpan):
    """Transparent annotation span created by @link_llm decorator.

    LLMScope marks a code boundary where LLM calls happen. It is transparent
    for parent attribution - LLMCalls reference it via llm_scope_id for
    grouping but get their parent_span_id from structural spans (Agent/Tool).

    Not included in get_llm_calls() or verify_llm() results.

    Attributes:
        provider: Provider specified in the decorator, or None for auto-detection.
            When None, the provider is inferred from HTTP interception.
        model: Model specified in the decorator, if any.
        caller_name: Function name where decorator was applied.
        caller_signature: Function signature where decorator was applied.
        caller_location: File:line location of the decorated function.
        input_data: Positional arguments passed to the decorated function.
        input_kwargs: Keyword arguments passed to the decorated function.
        output_data: Return value from the decorated function.
    """

    provider: str | None = None
    model: str | None = None
    caller_name: str | None = None
    caller_signature: str | None = None
    caller_location: str | None = None

    input_data: tuple[Any, ...] = Field(default_factory=tuple)
    input_kwargs: dict[str, Any] = Field(default_factory=dict)
    output_data: Any = None


class ToolCall(BaseSpan):
    """Mutable object representing a tool call lifecycle.

    Created when tool is invoked, updated in-place when completed.
    Tracks arguments, responses, timing, and errors.

    Attributes:
        target_path: Fully qualified path for verification matching
            (e.g., "mymod.search_tool"). Used by verify_tool() to match spans.
        display_name: Human-readable name of the tool (for display/trace output).
        args: Positional arguments passed to the tool.
        kwargs: Keyword arguments passed to the tool.
        result: Tool response data, if any.
        simulated: Whether this call was intercepted by simulation.
    """

    target_path: str
    display_name: str | None = None
    args: tuple[Any, ...] = Field(default_factory=tuple)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    simulated: bool = False


class AgentRun(BaseSpan):
    """Mutable object representing an agent execution lifecycle.

    Created when agent starts, updated in-place when completed.
    Represents the top-level span for agent operations, with child
    spans for LLM calls and tool calls.

    The `id` field is a unique span identifier for this specific run.
    The `agent_id` field equals `id` (an agent belongs to itself).

    Used for:
    - Multi-agent hierarchies (Manager -> Researcher -> Writer)
    - Recursive agent composition (Agent calls Agent)
    - Stack-based trace propagation across async boundaries

    Attributes:
        target_path: Fully qualified path for verification matching
            (e.g., "mymod.MyAgent.run"). Used by verify_agent() to match spans.
        display_name: Human-readable agent name (for display/trace output).
        parent_agent_id: Parent agent span identifier, if any.
        invoked_by_tool_call_id: ID of the ToolCall that spawned this agent, if any.
        spans: Child spans collected under this agent.
        input_data: Input payload for the agent, if provided.
        output_data: Output payload from the agent, if available.
        kwargs: Keyword arguments passed to the agent.
        simulated: Whether this call was intercepted by simulation.
    """

    target_path: str
    display_name: str | None = None
    parent_agent_id: str | None = None
    invoked_by_tool_call_id: str | None = None
    spans: list[BaseSpan] = Field(default_factory=list)
    input_data: Any = None
    output_data: Any = None
    kwargs: dict[str, Any] = Field(default_factory=dict)
    simulated: bool = False

    def get_llm_calls(self, recursive: bool = True) -> list[LLMCall]:
        """Get all LLM calls in this agent's execution.

        Args:
            recursive: Include LLM calls from nested agents.

        Returns:
            List of LLM calls.
        """
        llm_calls = [s for s in self.spans if isinstance(s, LLMCall)]

        if recursive:
            for child in self.spans:
                if isinstance(child, AgentRun):
                    llm_calls.extend(child.get_llm_calls(recursive=True))

        return llm_calls

    def get_tool_calls(self, recursive: bool = True) -> list[ToolCall]:
        """Get all tool calls in this agent's execution.

        Args:
            recursive: Include tool calls from nested agents.

        Returns:
            List of tool calls.
        """
        tool_calls = [s for s in self.spans if isinstance(s, ToolCall)]

        if recursive:
            for child in self.spans:
                if isinstance(child, AgentRun):
                    tool_calls.extend(child.get_tool_calls(recursive=True))

        return tool_calls

    def get_child_agents(self, recursive: bool = True) -> list[AgentRun]:
        """Get all nested agents in this agent's execution.

        Args:
            recursive: Include deeply nested agents.

        Returns:
            List of child agent runs.
        """
        child_agents = [s for s in self.spans if isinstance(s, AgentRun)]

        if recursive:
            for child in child_agents[:]:  # Copy to avoid mutation during iteration
                child_agents.extend(child.get_child_agents(recursive=True))

        return child_agents
