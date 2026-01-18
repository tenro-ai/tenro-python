# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Context management for span stack tracking.

Provides async-safe isolation for span stacks so concurrent agent runs do not
interfere with each other.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tenro._core.spans import BaseSpan, LLMScope

_current_span: ContextVar[BaseSpan | None] = ContextVar("current_span", default=None)
_span_stack: ContextVar[list[BaseSpan] | None] = ContextVar("span_stack", default=None)


def _get_stack() -> list[BaseSpan]:
    """Get current span stack, initializing if needed."""
    stack = _span_stack.get(None)
    if stack is None:
        stack = []
        _span_stack.set(stack)
    return stack


def push_span(span: BaseSpan) -> None:
    """Push span onto context stack.

    Sets span as current context and adds to stack for parent tracking.
    Thread-safe and async-safe via contextvars. Creates a new list to ensure
    async task isolation.

    Args:
        span: The span to push (LLMCall, ToolCall, or AgentRun).
    """
    stack = _get_stack()
    new_stack = [*stack, span]
    _span_stack.set(new_stack)
    _current_span.set(span)


def pop_span() -> BaseSpan | None:
    """Pop span from context stack.

    Removes span from stack and updates current context to parent.
    Returns the popped span for verification.

    Creates a new list to ensure async task isolation.

    Returns:
        The popped span, or `None` if stack was empty.
    """
    stack = _get_stack()
    if not stack:
        _current_span.set(None)
        return None

    popped = stack[-1]
    new_stack = stack[:-1]
    _span_stack.set(new_stack)

    _current_span.set(new_stack[-1] if new_stack else None)

    return popped


def get_current_span() -> BaseSpan | None:
    """Get currently active span.

    Returns the span at the top of the stack, or `None` if no span is active.
    Used for automatic parent_id tracking.

    Returns:
        Current span, or `None` if stack is empty.
    """
    return _current_span.get(None)


def get_trace_id() -> str | None:
    """Get current trace ID from active span.

    Returns:
        Trace ID from current span, or `None` if no span is active.
    """
    current = get_current_span()
    return current.trace_id if current else None


def get_span_stack() -> list[BaseSpan]:
    """Get full span stack for debugging.

    Returns a copy of the stack to prevent external mutation.

    Returns:
        Copy of current span stack (may be empty).
    """
    return _get_stack().copy()


def clear_context() -> None:
    """Clear span context.

    Used for cleanup in tests or error recovery.
    """
    _span_stack.set([])
    _current_span.set(None)


def get_current_agent_name() -> str | None:
    """Get name of current agent from span stack.

    Walks the span stack from top to bottom looking for an AgentRun span.
    Used by HTTP interceptor for LLM call attribution.

    Returns:
        Name of the nearest AgentRun span, or `None` if no agent is active.
    """
    from tenro._core.spans import AgentRun

    for span in reversed(_get_stack()):
        if isinstance(span, AgentRun):
            return span.display_name or span.target_path
    return None


def get_nearest_llm_scope() -> LLMScope | None:
    """Get nearest LLMScope from span stack.

    Walks the span stack from top to bottom looking for an LLMScope span.
    Used by handle_http_call() to set llm_scope_id on LLMCall spans.

    LLMScope is transparent for parent attribution - this function is only
    used for grouping metadata, not structural parent linking.

    Returns:
        Nearest LLMScope span, or `None` if no scope is active.
    """
    from tenro._core.spans import LLMScope

    for span in reversed(_get_stack()):
        if isinstance(span, LLMScope):
            return span
    return None
