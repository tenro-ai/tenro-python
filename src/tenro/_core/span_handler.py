# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""SpanHandler protocol and implementations for span lifecycle management.

Provides pluggable span event emission via a protocol that LifecycleManager
delegates to after mutating span state (error, status, latency).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from tenro._core.spans import BaseSpan

logger = logging.getLogger(__name__)


@runtime_checkable
class SpanHandler(Protocol):
    """Protocol for span lifecycle handlers.

    Implementations receive lifecycle callbacks from LifecycleManager.
    The span is already mutated (error/status/latency set) before
    handler methods are called.
    """

    def on_span_start(
        self,
        span: BaseSpan,
        *,
        parent_id: str | None,
        start_time_ns: int | None = None,
    ) -> None:
        """Called when a span starts.

        Args:
            span: The span that started (context already set up).
            parent_id: Structural parent span ID, or None for root.
            start_time_ns: Optional nanosecond timestamp override.
        """
        ...

    def on_span_update(self, span: BaseSpan) -> None:
        """Called when span data changes mid-flight.

        Args:
            span: The span with updated fields.
        """
        ...

    def on_span_error(
        self,
        span: BaseSpan,
        *,
        parent_id: str | None,
        error: Exception,
    ) -> None:
        """Called when a span encounters an error.

        Args:
            span: The span (error/status already set by LifecycleManager).
            parent_id: Structural parent span ID, or None for root.
            error: The exception that occurred.
        """
        ...

    def on_span_end(
        self,
        span: BaseSpan,
        *,
        parent_id: str | None,
        end_time_ns: int | None = None,
    ) -> None:
        """Called when a span ends (both success and error paths).

        Args:
            span: The finalized span (latency/status set).
            parent_id: Structural parent span ID, or None for root.
            end_time_ns: Optional nanosecond timestamp override.
        """
        ...


class CompositeSpanHandler:
    """Dispatches lifecycle calls to primary + secondary handlers.

    Primary handler (eval) errors propagate. Secondary handler (OTel)
    errors are caught and logged — export failures never break evaluation.
    """

    def __init__(
        self,
        primary: SpanHandler,
        secondary: list[SpanHandler],
    ) -> None:
        self._primary = primary
        self._secondary = secondary

    def on_span_start(
        self,
        span: BaseSpan,
        *,
        parent_id: str | None,
        start_time_ns: int | None = None,
    ) -> None:
        self._primary.on_span_start(span, parent_id=parent_id, start_time_ns=start_time_ns)
        for backend in self._secondary:
            _call_safe(
                backend.on_span_start,
                span,
                parent_id=parent_id,
                start_time_ns=start_time_ns,
            )

    def on_span_update(self, span: BaseSpan) -> None:
        self._primary.on_span_update(span)
        for backend in self._secondary:
            _call_safe(backend.on_span_update, span)

    def on_span_error(
        self,
        span: BaseSpan,
        *,
        parent_id: str | None,
        error: Exception,
    ) -> None:
        self._primary.on_span_error(span, parent_id=parent_id, error=error)
        for backend in self._secondary:
            _call_safe(backend.on_span_error, span, parent_id=parent_id, error=error)

    def on_span_end(
        self,
        span: BaseSpan,
        *,
        parent_id: str | None,
        end_time_ns: int | None = None,
    ) -> None:
        self._primary.on_span_end(span, parent_id=parent_id, end_time_ns=end_time_ns)
        for backend in self._secondary:
            _call_safe(
                backend.on_span_end,
                span,
                parent_id=parent_id,
                end_time_ns=end_time_ns,
            )


def _call_safe(fn: Any, *args: Any, **kwargs: Any) -> None:
    """Call a secondary handler method with error isolation."""
    try:
        fn(*args, **kwargs)
    except Exception:
        logger.warning(
            "SpanHandler %s failed; eval unaffected",
            getattr(fn, "__qualname__", fn),
            exc_info=True,
        )
