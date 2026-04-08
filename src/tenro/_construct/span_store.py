# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe span storage and collection for completed spans."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tenro._core.spans import BaseSpan


class SpanStore:
    """Thread-safe store for completed span objects."""

    def __init__(self) -> None:
        """Initialize empty span store."""
        self._spans: list[BaseSpan] = []
        self._lock = threading.Lock()

    def store(self, span: BaseSpan) -> None:
        """Store a completed span (thread-safe).

        Args:
            span: Completed span to store.
        """
        with self._lock:
            self._spans.append(span)

    def get_all_spans(self) -> list[BaseSpan]:
        """Get all stored spans.

        Returns:
            Copy of all stored spans.
        """
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        """Clear all stored spans."""
        with self._lock:
            self._spans.clear()


class SpanCollector:
    """SpanHandler that stores complete spans on end.

    Span objects are mutated in-place by LifecycleManager,
    then stored here on completion.
    """

    def __init__(self, span_store: SpanStore) -> None:
        """Initialize with span store.

        Args:
            span_store: Store for completed spans.
        """
        self._store = span_store

    def on_span_start(
        self,
        span: BaseSpan,
        *,
        parent_id: str | None,
        start_time_ns: int | None = None,
    ) -> None:
        """Span start is handled in memory by LifecycleManager."""

    def on_span_update(self, span: BaseSpan) -> None:
        """Span updates are handled in memory by LifecycleManager."""

    def on_span_error(
        self,
        span: BaseSpan,
        *,
        parent_id: str | None,
        error: Exception,
    ) -> None:
        """Error already set on span by LifecycleManager."""

    def on_span_end(
        self,
        span: BaseSpan,
        *,
        parent_id: str | None,
        end_time_ns: int | None = None,
    ) -> None:
        """Store the completed span.

        Args:
            span: Finalized span (latency/status set).
            parent_id: Structural parent span ID.
            end_time_ns: Optional nanosecond timestamp override.
        """
        self._store.store(span)


__all__ = ["SpanCollector", "SpanStore"]
