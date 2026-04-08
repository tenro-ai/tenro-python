# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle management for span tracking.

Context manager that handles span creation, parent linking, and delegates
to a SpanHandler for storage and export.
"""

from __future__ import annotations

import bisect
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

from tenro._core import context
from tenro._core.spans import AgentRun, BaseSpan, LLMCall, LLMScope, SpanContext
from tenro.errors import TenroAgentRecursionError

if TYPE_CHECKING:
    from collections.abc import Generator

    from tenro._core.span_handler import SpanHandler


class LifecycleManager:
    """Manages span lifecycle with handler-based tracking.

    Tracks spans from start to completion, notifies a SpanHandler at each
    lifecycle point, and maintains parent-child relationships via a context
    stack. Enforces depth limits to prevent infinite recursion.
    """

    def __init__(
        self,
        handler: SpanHandler,
        max_depth: int = 25,
    ) -> None:
        """Initialize lifecycle manager with a span handler.

        Args:
            handler: Handler for span lifecycle events.
            max_depth: Maximum nesting depth before raising an error.
        """
        self._handler = handler
        self._max_depth = max_depth
        self._creation_seq = 0

    @contextmanager
    def start_span(
        self,
        span: BaseSpan,
        remote_parent: SpanContext | None = None,
    ) -> Generator[BaseSpan, None, None]:
        """Start and manage a span's lifecycle.

        This context manager provides stack-safe lifecycle management:
        - Enforces depth limits to prevent infinite loops
        - Auto-links agent_id from current agent context
        - Supports remote parent from trace propagation
        - Notifies handler of span start
        - Pushes to context stack
        - Yields span for updates during the call
        - Handles exceptions and notifies handler
        - Notifies handler of span end
        - Pops from stack even when exceptions occur

        Args:
            span: The span to manage (LLMCall, ToolCall, or AgentRun).
            remote_parent: Extracted remote parent context from propagation.
                Used when no local parent exists on the context stack.

        Yields:
            The span object for tracking state (not persisted).

        Raises:
            AgentRecursionError: If depth limit exceeded.
        """
        self._check_depth()
        parent_span_id = self._setup_span_context(span, remote_parent)
        self._handler.on_span_start(span, parent_id=parent_span_id)
        context.push_span(span)

        try:
            yield span
        except Exception as e:
            self._handle_error(span, parent_span_id, e)
            raise
        finally:
            self._pop_expected(span)
            self._finalize_span(span, parent_span_id)

    def start_span_manual(
        self,
        span: BaseSpan,
        remote_parent: SpanContext | None = None,
    ) -> str | None:
        """Start a span without using a context manager.

        Use this for generators where the span must stay open during iteration.
        Caller must call `end_span_manual` or `error_span_manual` to close.

        Args:
            span: The span to start.
            remote_parent: Extracted remote parent context from propagation.

        Returns:
            Parent span ID to pass to end/error calls.

        Raises:
            AgentRecursionError: If depth limit is exceeded.
        """
        self._check_depth()
        parent_span_id = self._setup_span_context(span, remote_parent)
        self._handler.on_span_start(span, parent_id=parent_span_id)
        context.push_span(span)
        return parent_span_id

    def end_span_manual(self, span: BaseSpan, parent_span_id: str | None) -> None:
        """End a span that was started with `start_span_manual`.

        Marks the span as completed and notifies handler.

        Args:
            span: The span to end.
            parent_span_id: Value returned from `start_span_manual`.

        Raises:
            RuntimeError: If span is not on top of the context stack (LIFO violation).
        """
        self._pop_expected(span)
        self._finalize_span(span, parent_span_id)

    def error_span_manual(
        self, span: BaseSpan, parent_span_id: str | None, error: Exception
    ) -> None:
        """Record an error for a span started with `start_span_manual`.

        Marks the span as failed, notifies handler of error, then finalizes
        the span so backends receive both on_span_error and on_span_end.

        Args:
            span: The span that encountered an error.
            parent_span_id: Value returned from `start_span_manual`.
            error: The exception that occurred.

        Raises:
            RuntimeError: If span is not on top of the context stack (LIFO violation).
        """
        self._pop_expected(span)
        self._handle_error(span, parent_span_id, error)
        self._finalize_span(span, parent_span_id)

    def _pop_expected(self, expected: BaseSpan) -> None:
        """Pop span from stack, verifying LIFO order.

        Checks before popping so the stack stays intact on mismatch.

        Raises RuntimeError if the top span doesn't match, which
        indicates out-of-order closes or cross-task stack corruption.
        """
        current = context.get_current_span()
        if current is not expected:
            raise RuntimeError(
                f"Stack integrity violation: expected to pop {expected.span_id!r} "
                f"but got {current.span_id if current else 'empty stack'!r}. "
                "This indicates out-of-order span closes."
            )
        context.pop_span()

    def _check_depth(self) -> None:
        """Check depth limit and raise if exceeded."""
        stack = context.get_span_stack()
        if len(stack) >= self._max_depth:
            raise TenroAgentRecursionError(
                f"Trace depth exceeded limit ({self._max_depth}). "
                "This usually indicates an infinite loop between agents."
            )

    def _setup_span_context(
        self, span: BaseSpan, remote_parent: SpanContext | None = None
    ) -> str | None:
        """Set up span context from stack or remote parent.

        Priority: local context stack > remote parent > root span.

        Sets both:
        - parent_span_id: Structural parent (skips LLMScope which is transparent)
        - agent_id: Closest agent ancestor (for agent-centric queries)

        LLMScope is transparent for parent attribution. When the current span
        is LLMScope, the structural parent is used instead (Agent, Tool, LLM).
        """
        span.creation_seq = self._creation_seq
        self._creation_seq += 1

        stack = context.get_span_stack()
        current = context.get_current_span()

        # Invariant: LLMCall inside LLMScope must have llm_scope_id set
        self._check_llm_scope_invariant(span, stack)

        # Path 1: Local parent exists on the context stack
        if current:
            span.trace_id = current.trace_id
            span.trace_flags = current.trace_flags
            span.trace_state = current.trace_state
            structural_parent = self._find_structural_parent(stack)
            span.parent_span_id = structural_parent.span_id if structural_parent else None
            agent_ancestor = self._find_agent_ancestor(stack)
            self._link_to_parent(span, agent_ancestor)
            return structural_parent.span_id if structural_parent else None

        # Path 2: Remote parent from trace propagation (no local context)
        if remote_parent is not None:
            span.trace_id = remote_parent.trace_id
            span.parent_span_id = remote_parent.span_id
            span.trace_flags = remote_parent.trace_flags
            span.trace_state = remote_parent.trace_state
            span.parent_is_remote = True
            if isinstance(span, AgentRun):
                span.agent_id = span.span_id
            return remote_parent.span_id

        # Path 3: Root span (no parent)
        if isinstance(span, AgentRun):
            span.agent_id = span.span_id
        return None

    def _find_structural_parent(self, stack: list[BaseSpan]) -> BaseSpan | None:
        """Find closest structural parent, skipping LLMScope.

        LLMScope is transparent for parent attribution - LLMCalls should
        reference their structural parent (Agent, Tool, or LLM), not the scope.
        """
        for ancestor in reversed(stack):
            if not isinstance(ancestor, LLMScope):
                return ancestor
        return None

    def _find_agent_ancestor(self, stack: list[BaseSpan]) -> AgentRun | None:
        """Find closest AgentRun in stack."""
        for ancestor in reversed(stack):
            if isinstance(ancestor, AgentRun):
                return ancestor
        return None

    def _check_llm_scope_invariant(self, span: BaseSpan, stack: list[BaseSpan]) -> None:
        """Fail fast if LLMCall inside LLMScope has wrong llm_scope_id.

        Validates both presence and correctness: llm_scope_id must match
        the nearest enclosing LLMScope on the stack.
        """
        if not isinstance(span, LLMCall):
            return

        nearest_scope: LLMScope | None = None
        for s in reversed(stack):
            if isinstance(s, LLMScope):
                nearest_scope = s
                break

        if nearest_scope is not None and span.llm_scope_id is None:
            raise RuntimeError(
                "Invariant violation: LLMCall created inside LLMScope context "
                "but llm_scope_id is not set. This indicates a bug in span creation - "
                "all LLMCall creation paths must use _create_llm_span() helper."
            )

        if (
            nearest_scope is not None
            and span.llm_scope_id is not None
            and span.llm_scope_id != nearest_scope.span_id
        ):
            raise RuntimeError(
                f"Invariant violation: LLMCall.llm_scope_id={span.llm_scope_id!r} "
                f"does not match nearest LLMScope.span_id={nearest_scope.span_id!r}. "
                "This indicates a bug in span creation."
            )

    def _link_to_parent(self, span: BaseSpan, agent: AgentRun | None) -> None:
        """Set agent_id and parent_agent_id from nearest agent ancestor."""
        if isinstance(span, AgentRun):
            span.agent_id = span.span_id
            if agent:
                span.parent_agent_id = agent.span_id
        elif agent:
            span.agent_id = agent.span_id

    def _handle_error(self, span: BaseSpan, parent_span_id: str | None, error: Exception) -> None:
        """Handle error: mutate span state and notify handler."""
        span.error = str(error) or type(error).__name__
        span.status_code = "error"
        span.status_message = str(error) or type(error).__name__
        self._handler.on_span_error(span, parent_id=parent_span_id, error=error)

    def _finalize_span(self, span: BaseSpan, parent_span_id: str | None) -> None:
        """Finalize span: set end_time/status, link to parent, and notify handler."""
        span.end_time = time.time_ns()
        self._link_to_parent_agent(span)
        self._handler.on_span_end(span, parent_id=parent_span_id)

    def _link_to_parent_agent(self, span: BaseSpan) -> None:
        """Add completed span to nearest agent ancestor's .spans list.

        Uses bisect.insort to maintain start_time order. Spans complete
        inner-first (nested), but users expect creation order.
        """
        current = context.get_current_span()
        if current is None:
            return
        # Walk stack from top to find nearest AgentRun
        for ancestor in reversed(context.get_span_stack()):
            if isinstance(ancestor, AgentRun):
                bisect.insort(  # type: ignore[misc]
                    ancestor.spans,
                    span,
                    key=lambda s: (s.start_time, s.creation_seq),
                )
                return
