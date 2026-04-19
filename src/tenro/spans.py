# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Span models for tracking LLM calls, tool calls, and agent runs.

Spans track operation lifecycle during test execution, updating in-place
as operations complete.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, computed_field, field_validator, model_validator

from tenro._core.model_base import BaseModel
from tenro._core.spans import (
    _validate_optional_span_id,
    _validate_span_id,
    _validate_trace_flags,
    _validate_trace_id,
    _validate_trace_state,
)

SpanType = Literal["llm", "tool", "agent", "llm_scope"]
SpanKind = Literal["CLIENT", "SERVER", "INTERNAL", "PRODUCER", "CONSUMER"]
StatusCode = Literal["unset", "error", "ok"]


class SpanContext(PydanticBaseModel):
    """Immutable W3C trace context for a span.

    Carries the identifiers needed for trace propagation and span linking.

    Attributes:
        trace_id: 32 lowercase hex chars (16 bytes), non-zero.
        span_id: 16 lowercase hex chars (8 bytes), non-zero.
        trace_flags: 2 lowercase hex chars (1 byte). Stored as the hex wire
            format string, not the numeric ``TraceFlags(int)`` representation.
        trace_state: Raw W3C tracestate header string, or None. Stored as
            the unparsed header value, not a structured key-value type.
        is_remote: True when this context was extracted from propagation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    span_id: str
    trace_flags: str = "00"
    trace_state: str | None = None
    is_remote: bool = False

    _validate_trace_id = field_validator("trace_id")(_validate_trace_id)
    _validate_span_id = field_validator("span_id")(_validate_span_id)
    _validate_trace_flags = field_validator("trace_flags")(_validate_trace_flags)
    _validate_trace_state = field_validator("trace_state")(_validate_trace_state)


class SpanLink(PydanticBaseModel):
    """Link to another span's context.

    Used for retries, fan-out, and async handoffs. Compatible with
    OpenTelemetry span-link semantics.

    Note: ``attributes`` is only shallow-frozen — Pydantic's ``frozen=True``
    prevents reassigning the field, but the dict itself remains mutable.
    This is a known Pydantic limitation, acceptable for the testing SDK.

    Attributes:
        context: Full span context of the linked span.
        attributes: Additional link attributes (shallow-frozen).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    context: SpanContext
    attributes: dict[str, Any] = Field(default_factory=dict)


class BaseSpan(BaseModel):
    """Base class for all span types.

    Defines shared fields for timed operations in a trace.

    Attributes:
        span_id: Unique span identifier (16 hex chars).
        trace_id: Trace identifier (32 hex chars).
        start_time: Unix nanoseconds (from ``time.time_ns()``) when the span started.
        parent_span_id: Immediate parent span ID (Agent, LLM, or Tool).
        agent_span_id: Closest agent ancestor span ID, if any.
        error: Error message if the span failed.
        span_type: Discriminator for span subclass.
        kind: Span kind. Mirrors OpenTelemetry ``SpanKind`` semantics.
        name: Span name.
        end_time: Unix nanoseconds when the span ended, or None if in-flight.
        status_code: Status code (``unset``, ``error``, or ``ok``).
        status_message: Human-readable status message, if any.
        attributes: Attribute map. Supports standard OpenTelemetry and
            GenAI semantic-convention keys.
        trace_flags: W3C trace flags as the hex wire format (2 chars). Stored
            as a string, not the numeric ``TraceFlags(int)`` representation.
        trace_state: Raw W3C tracestate header string, unparsed.
        parent_is_remote: Derived convenience flag set by LifecycleManager
            when the parent was extracted from trace propagation. The
            canonical source is ``SpanContext.is_remote`` on the parent.
        links: Span links to related spans.
    """

    span_id: str
    trace_id: str
    start_time: int

    parent_span_id: str | None = None
    agent_span_id: str | None = None
    error: str | None = None

    span_type: SpanType
    kind: SpanKind
    name: str
    end_time: int | None = None
    status_code: StatusCode = "unset"
    status_message: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    trace_flags: str = "00"
    trace_state: str | None = None
    parent_is_remote: bool = False
    links: list[SpanLink] = Field(default_factory=list)

    _validate_trace_id = field_validator("trace_id")(_validate_trace_id)
    _validate_span_id = field_validator("span_id")(_validate_span_id)
    _validate_trace_flags = field_validator("trace_flags")(_validate_trace_flags)
    _validate_trace_state = field_validator("trace_state")(_validate_trace_state)
    _validate_parent_span_id = field_validator("parent_span_id")(_validate_optional_span_id)
    _validate_agent_span_id = field_validator("agent_span_id")(_validate_optional_span_id)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def latency_ms(self) -> float | None:
        """Span duration in milliseconds, or None if span is still in-flight."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) / 1_000_000

    @model_validator(mode="after")
    def _check_end_time(self) -> Self:
        if self.end_time is not None and self.end_time < self.start_time:
            msg = f"end_time ({self.end_time}) < start_time ({self.start_time})"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_remote_parent(self) -> Self:
        if self.parent_is_remote and self.parent_span_id is None:
            msg = "parent_is_remote=True requires parent_span_id"
            raise ValueError(msg)
        return self


class LLMCallSpan(BaseSpan):
    """Represents an LLM API call lifecycle.

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

    response_model: str | None = None
    response_id: str | None = None
    finish_reasons: list[str] | None = None


class LLMScope(BaseSpan):
    """Transparent annotation span created by @link_llm decorator.

    LLMScope marks a code boundary where LLM calls happen. It is transparent
    for parent attribution - LLMCallSpans reference it via llm_scope_id for
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


class ToolCallSpan(BaseSpan):
    """Represents a tool call lifecycle.

    Created when tool is invoked, updated in-place when completed.
    Tracks arguments, results, timing, and errors.

    Attributes:
        target_path: Fully qualified path for verification matching
            (e.g., "mymod.search_tool"). Used by verify_tool() to match spans.
        display_name: Human-readable name of the tool (for display/trace output).
        args: Positional arguments passed to the tool.
        kwargs: Keyword arguments passed to the tool.
        result: Tool result payload, if any.
    """

    target_path: str
    display_name: str | None = None
    args: tuple[Any, ...] = Field(default_factory=tuple)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    result: Any = None

    tool_call_id: str | None = None


class AgentRunSpan(BaseSpan):
    """Represents an agent execution lifecycle.

    Created when agent starts, updated in-place when completed.
    Represents the top-level span for agent operations, with child
    spans for LLM calls and tool calls.

    The `span_id` field is a unique span identifier for this specific run.
    The `agent_span_id` field equals `span_id` (an agent belongs to itself).

    Used for:
    - Multi-agent hierarchies (Manager -> Researcher -> Writer)
    - Recursive agent composition (Agent calls Agent)
    - Stack-based trace propagation across async boundaries

    Attributes:
        target_path: Fully qualified path for verification matching
            (e.g., "mymod.MyAgent.run"). Used by verify_agent() to match spans.
        display_name: Human-readable agent name (for display/trace output).
        agent_id: Stable agent identifier that survives renames.
            Falls back to display_name, then target_path when not set.
        version: Agent version string for deploy/config regression tracking.
        parent_agent_id: Parent agent span identifier, if any.
        invoked_by_tool_call_id: ID of the ToolCallSpan that spawned this agent,
            if any.
        spans: Child spans collected under this agent.
        input_data: Input payload for the agent, if provided.
        output_data: Output payload from the agent, if available.
        kwargs: Keyword arguments passed to the agent.
    """

    target_path: str
    display_name: str | None = None
    agent_id: str | None = None
    version: str | None = None
    parent_agent_id: str | None = None
    invoked_by_tool_call_id: str | None = None
    spans: list[BaseSpan] = Field(default_factory=list)
    input_data: Any = None
    output_data: Any = None
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_agent_span_id(self) -> Self:
        """An agent belongs to itself: agent_span_id defaults to span_id."""
        if self.agent_span_id is None:
            self.agent_span_id = self.span_id
        return self

    def get_llm_calls(self, recursive: bool = True) -> list[LLMCallSpan]:
        """Get all LLM calls in this agent's execution.

        Args:
            recursive: Include LLM calls from nested agents.

        Returns:
            List of LLM calls.
        """
        llm_calls = [s for s in self.spans if isinstance(s, LLMCallSpan)]

        if recursive:
            for child in self.spans:
                if isinstance(child, AgentRunSpan):
                    llm_calls.extend(child.get_llm_calls(recursive=True))

        return llm_calls

    def get_tool_calls(self, recursive: bool = True) -> list[ToolCallSpan]:
        """Get all tool calls in this agent's execution.

        Args:
            recursive: Include tool calls from nested agents.

        Returns:
            List of tool calls.
        """
        tool_calls = [s for s in self.spans if isinstance(s, ToolCallSpan)]

        if recursive:
            for child in self.spans:
                if isinstance(child, AgentRunSpan):
                    tool_calls.extend(child.get_tool_calls(recursive=True))

        return tool_calls

    def get_child_agents(self, recursive: bool = True) -> list[AgentRunSpan]:
        """Get all nested agents in this agent's execution.

        Args:
            recursive: Include deeply nested agents.

        Returns:
            List of child agent runs.
        """
        child_agents = [s for s in self.spans if isinstance(s, AgentRunSpan)]

        if recursive:
            for child in child_agents[:]:  # Copy to avoid mutation during iteration
                child_agents.extend(child.get_child_agents(recursive=True))

        return child_agents


__all__ = [
    "AgentRunSpan",
    "BaseSpan",
    "LLMCallSpan",
    "LLMScope",
    "SpanContext",
    "SpanKind",
    "SpanLink",
    "SpanType",
    "StatusCode",
    "ToolCallSpan",
]
