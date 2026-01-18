# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle management for span tracking.

Context manager that handles span creation, parent linking, and event emission
for LLM calls, tool calls, and agent runs.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

from tenro._core import context
from tenro._core.events import SpanEvent
from tenro._core.spans import AgentRun, BaseSpan, LLMCall, LLMScope, ToolCall
from tenro.errors import TenroAgentRecursionError

if TYPE_CHECKING:
    from collections.abc import Generator

    from tenro.protocols import EventStore


class LifecycleManager:
    """Manages span lifecycle with event-based tracking.

    Tracks spans from start to completion, emits events to the event store,
    and maintains parent-child relationships via a context stack. Enforces
    depth limits to prevent infinite recursion.
    """

    def __init__(
        self,
        event_store: EventStore,
        max_depth: int = 25,
    ) -> None:
        """Initialize lifecycle manager with event store.

        Args:
            event_store: Store for emitting span events.
            max_depth: Maximum nesting depth before raising an error.
        """
        self._event_store = event_store
        self._max_depth = max_depth

    @contextmanager
    def start_span(self, span: BaseSpan) -> Generator[BaseSpan, None, None]:
        """Start and manage a span's lifecycle with event-based tracking.

        This context manager provides stack-safe lifecycle management:
        - Enforces depth limits to prevent infinite loops
        - Auto-links agent_id from current agent context
        - Emits immutable start event
        - Pushes to context stack
        - Yields span for updates during the call
        - Handles exceptions and emits error event
        - Emits end event with final state
        - Pops from stack even when exceptions occur

        Args:
            span: The span to manage (LLMCall, ToolCall, or AgentRun).

        Yields:
            The span object for tracking state (not persisted).

        Raises:
            AgentRecursionError: If depth limit exceeded.
        """
        self._check_depth()
        parent_span_id = self._setup_span_context(span)
        self._emit_start_event(span, parent_span_id)
        context.push_span(span)

        try:
            yield span
        except Exception as e:
            self._handle_error(span, parent_span_id, e)
            raise
        finally:
            context.pop_span()
            self._finalize_span(span, parent_span_id)

    def start_span_manual(self, span: BaseSpan) -> str | None:
        """Start a span without using a context manager.

        Use this for generators where the span must stay open during iteration.
        Caller must call `end_span_manual` or `error_span_manual` to close.

        Args:
            span: The span to start.

        Returns:
            Parent span ID to pass to end/error calls.

        Raises:
            AgentRecursionError: If depth limit is exceeded.
        """
        self._check_depth()
        parent_span_id = self._setup_span_context(span)
        self._emit_start_event(span, parent_span_id)
        context.push_span(span)
        return parent_span_id

    def end_span_manual(self, span: BaseSpan, parent_span_id: str | None) -> None:
        """End a span that was started with `start_span_manual`.

        Marks the span as completed and emits the end event.

        Args:
            span: The span to end.
            parent_span_id: Value returned from `start_span_manual`.
        """
        context.pop_span()
        self._finalize_span(span, parent_span_id)

    def error_span_manual(
        self, span: BaseSpan, parent_span_id: str | None, error: Exception
    ) -> None:
        """Record an error for a span started with `start_span_manual`.

        Marks the span as failed and emits the error event.

        Args:
            span: The span that encountered an error.
            parent_span_id: Value returned from `start_span_manual`.
            error: The exception that occurred.
        """
        context.pop_span()
        self._handle_error(span, parent_span_id, error)

    def _check_depth(self) -> None:
        """Check depth limit and raise if exceeded."""
        stack = context.get_span_stack()
        if len(stack) >= self._max_depth:
            raise TenroAgentRecursionError(
                f"Trace depth exceeded limit ({self._max_depth}). "
                "This usually indicates an infinite loop between agents."
            )

    def _setup_span_context(self, span: BaseSpan) -> str | None:
        """Setup span context from stack and return parent_span_id.

        Sets both:
        - parent_span_id: Structural parent (skips LLMScope which is transparent)
        - agent_id: Closest agent ancestor (for agent-centric queries)

        LLMScope is transparent for parent attribution. When the current span
        is LLMScope, the structural parent is used instead (Agent, Tool, LLM).
        """
        stack = context.get_span_stack()
        current = context.get_current_span()

        # Invariant: LLMCall inside LLMScope must have llm_scope_id set
        self._check_llm_scope_invariant(span, stack)

        if current:
            span.trace_id = current.trace_id
            # Skip LLMScope for structural parent - it's transparent
            structural_parent = self._find_structural_parent(stack)
            span.parent_span_id = structural_parent.id if structural_parent else None
            agent_ancestor = self._find_agent_ancestor(stack)
            self._link_to_parent(span, agent_ancestor)
            return structural_parent.id if structural_parent else None

        if isinstance(span, AgentRun):
            span.agent_id = span.id
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
        """Fail fast if LLMCall inside LLMScope lacks llm_scope_id.

        This invariant catches bugs where a new LLMCall creation path
        forgets to look up the enclosing LLMScope context.
        """
        if not isinstance(span, LLMCall):
            return

        has_llm_scope = any(isinstance(s, LLMScope) for s in stack)
        if has_llm_scope and span.llm_scope_id is None:
            raise RuntimeError(
                "Invariant violation: LLMCall created inside LLMScope context "
                "but llm_scope_id is not set. This indicates a bug in span creation - "
                "all LLMCall creation paths must use _create_llm_span() helper."
            )

    def _link_to_parent(self, span: BaseSpan, agent: AgentRun | None) -> str | None:
        """Link span to parent and return parent_span_id."""
        if isinstance(span, AgentRun):
            span.agent_id = span.id
            if agent:
                span.parent_agent_id = agent.id
                return agent.id
        elif agent:
            span.agent_id = agent.id
            return agent.id
        return None

    def _emit_start_event(self, span: BaseSpan, parent_span_id: str | None) -> None:
        """Emit start event for span."""
        self._event_store.emit(
            SpanEvent.create_start(
                trace_id=span.trace_id,
                span_id=span.id,
                parent_span_id=parent_span_id,
                span_kind=self._get_span_kind(span),
                data=self._extract_start_payload(span),
                order_index=self._event_store.get_next_order_index(),
            )
        )

    def _handle_error(self, span: BaseSpan, parent_span_id: str | None, error: Exception) -> None:
        """Handle error and emit error event."""
        span.error = str(error)
        span.status = "error"
        self._event_store.emit(
            SpanEvent.create_error(
                trace_id=span.trace_id,
                span_id=span.id,
                parent_span_id=parent_span_id,
                span_kind=self._get_span_kind(span),
                error_message=str(error),
                order_index=self._event_store.get_next_order_index(),
            )
        )

    def _finalize_span(self, span: BaseSpan, parent_span_id: str | None) -> None:
        """Finalize span and emit end event if successful."""
        span.latency_ms = (time.time() - span.start_time) * 1000
        if not span.error:
            span.status = "completed"
            self._event_store.emit(
                SpanEvent.create_end(
                    trace_id=span.trace_id,
                    span_id=span.id,
                    parent_span_id=parent_span_id,
                    span_kind=self._get_span_kind(span),
                    data=self._extract_end_payload(span),
                    order_index=self._event_store.get_next_order_index(),
                )
            )

    def _get_span_kind(self, span: BaseSpan) -> Literal["AGENT", "LLM", "TOOL", "LLM_SCOPE"]:
        """Get span kind for event."""
        if isinstance(span, AgentRun):
            return "AGENT"
        elif isinstance(span, LLMScope):
            return "LLM_SCOPE"
        elif isinstance(span, LLMCall):
            return "LLM"
        elif isinstance(span, ToolCall):
            return "TOOL"
        return "AGENT"

    def _extract_start_payload(self, span: BaseSpan) -> dict[str, Any]:
        """Extract start event payload from span."""
        if isinstance(span, LLMScope):
            return {
                "provider": span.provider,
                "model": span.model,
                "caller_name": span.caller_name,
                "caller_signature": span.caller_signature,
                "caller_location": span.caller_location,
                "input_data": list(span.input_data),
                "input_kwargs": span.input_kwargs,
            }
        elif isinstance(span, LLMCall):
            return {
                "provider": span.provider,
                "messages": span.messages,
                "model": span.model,
                "caller_signature": span.caller_signature,
                "caller_location": span.caller_location,
                "agent_name": span.agent_name,
                "llm_scope_id": span.llm_scope_id,
                "target_path": span.target_path,
            }
        elif isinstance(span, ToolCall):
            return {
                "tool_name": span.display_name,
                "target_path": span.target_path,
                "args": list(span.args),
                "kwargs": span.kwargs,
            }
        elif isinstance(span, AgentRun):
            return {
                "agent_name": span.display_name,
                "target_path": span.target_path,
                "input_data": span.input_data,
                "kwargs": span.kwargs,
            }
        return {}

    def _extract_end_payload(self, span: BaseSpan) -> dict[str, Any]:
        """Extract end event payload from span."""
        if isinstance(span, LLMScope):
            return {
                "output_data": span.output_data,
            }
        elif isinstance(span, LLMCall):
            return {
                "text": span.response or "",
                "usage": span.token_usage,
                "model": span.model,
                "tool_calls": span.tool_calls,
                "simulated": span.simulated,
            }
        elif isinstance(span, ToolCall):
            return {
                "args": list(span.args),
                "kwargs": span.kwargs,
                "result": span.result,
                "simulated": span.simulated,
            }
        elif isinstance(span, AgentRun):
            return {
                "output_data": span.output_data,
                "simulated": span.simulated,
            }
        return {}
