# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Span factory functions for decorator-based linking."""

from __future__ import annotations

import time
from typing import Any

from tenro._core.spans import (
    AgentRun,
    LLMCall,
    LLMScope,
    ToolCall,
    _generate_span_id,
    _generate_trace_id,
)


def create_agent_span(
    target_path: str,
    display_name: str,
    input_data: Any = None,
    kwargs: dict[str, Any] | None = None,
    *,
    agent_id: str | None = None,
    version: str | None = None,
) -> AgentRun:
    """Create an AgentRun span.

    Args:
        target_path: Fully qualified path (e.g., "mymod.Agent.run").
        display_name: Human-readable agent name.
        input_data: Input data for the agent.
        kwargs: Keyword arguments passed to the agent.
        agent_id: Stable agent identifier that survives renames.
        version: Agent version string for regression tracking.

    Returns:
        Initialized AgentRun span.
    """
    return AgentRun(
        span_id=_generate_span_id(),
        trace_id=_generate_trace_id(),
        start_time=time.time_ns(),
        span_type="agent",
        kind="INTERNAL",
        name=f"invoke_agent {display_name}",
        target_path=target_path,
        display_name=display_name,
        agent_id=agent_id,
        version=version,
        input_data=input_data,
        kwargs=kwargs or {},
    )


def create_tool_span(
    target_path: str,
    display_name: str,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> ToolCall:
    """Create a ToolCall span.

    Args:
        target_path: Fully qualified path (e.g., "mymod.search").
        display_name: Human-readable tool name.
        args: Positional arguments passed to the tool.
        kwargs: Keyword arguments passed to the tool.

    Returns:
        Initialized ToolCall span.
    """
    return ToolCall(
        span_id=_generate_span_id(),
        trace_id=_generate_trace_id(),
        start_time=time.time_ns(),
        span_type="tool",
        kind="INTERNAL",
        name=f"execute_tool {display_name}",
        target_path=target_path,
        display_name=display_name,
        args=args,
        kwargs=kwargs or {},
    )


def create_llm_scope_span(
    caller_name: str,
    provider: str | None = None,
    model: str | None = None,
    caller_signature: str | None = None,
    caller_location: str | None = None,
    input_data: tuple[Any, ...] = (),
    input_kwargs: dict[str, Any] | None = None,
) -> LLMScope:
    """Create an LLMScope span.

    Args:
        caller_name: Function name where decorator was applied.
        provider: Provider specified in the decorator.
        model: Model specified in the decorator.
        caller_signature: Function signature string.
        caller_location: File:line location of the decorated function.
        input_data: Positional arguments passed to the decorated function.
        input_kwargs: Keyword arguments passed to the decorated function.

    Returns:
        Initialized LLMScope span.
    """
    return LLMScope(
        span_id=_generate_span_id(),
        trace_id=_generate_trace_id(),
        start_time=time.time_ns(),
        span_type="llm_scope",
        kind="INTERNAL",
        name=f"link_llm {caller_name}",
        provider=provider,
        model=model,
        caller_name=caller_name,
        caller_signature=caller_signature,
        caller_location=caller_location,
        input_data=input_data,
        input_kwargs=input_kwargs or {},
    )


def create_llm_call_span(
    *,
    provider: str,
    model: str | None = None,
    target_path: str | None = None,
    llm_scope_id: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    response: str | None = None,
) -> LLMCall:
    """Create an LLMCall span.

    Args:
        provider: LLM provider name (e.g., "openai").
        model: Model identifier.
        target_path: Target function path for the LLM call.
        llm_scope_id: ID of enclosing LLMScope.
        messages: Messages sent to the provider.
        response: Model response text.

    Returns:
        Initialized LLMCall span.
    """
    return LLMCall(
        span_id=_generate_span_id(),
        trace_id=_generate_trace_id(),
        start_time=time.time_ns(),
        span_type="llm",
        kind="CLIENT",
        name=f"chat {model or provider}",
        provider=provider,
        messages=messages or [],
        response=response,
        model=model,
        target_path=target_path,
        llm_scope_id=llm_scope_id,
    )
