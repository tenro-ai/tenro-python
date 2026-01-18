# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""In-memory event store for span events.

Append-only storage for immutable span events.
Thread-safe for parallel agent execution.
"""

from __future__ import annotations

import threading

from tenro._core.events import SpanEvent


class EventStore:
    """In-memory event store with thread-safe operations.

    Stores events in an append-only list for deterministic replay.
    """

    def __init__(self) -> None:
        """Initialize empty event store."""
        self._events: list[SpanEvent] = []
        self._lock = threading.Lock()
        self._order_counter = 0

    def emit(self, event: SpanEvent) -> None:
        """Append event to store (thread-safe).

        Args:
            event: Event to persist.
        """
        with self._lock:
            self._events.append(event)

    def get_events(self, trace_id: str) -> list[SpanEvent]:
        """Retrieve all events for a trace, sorted chronologically.

        Args:
            trace_id: Trace ID to filter by.

        Returns:
            Events sorted by (timestamp, order_index).
        """
        with self._lock:
            filtered = [e for e in self._events if e.trace_id == trace_id]
            return sorted(filtered, key=lambda e: (e.timestamp, e.order_index))

    def get_all_events(self) -> list[SpanEvent]:
        """Retrieve all events across all traces.

        Returns:
            All events sorted by (timestamp, order_index).
        """
        with self._lock:
            return sorted(self._events, key=lambda e: (e.timestamp, e.order_index))

    def get_next_order_index(self) -> int:
        """Get next order index for same-millisecond event ordering.

        Returns:
            Incrementing integer for order_index field.
        """
        with self._lock:
            self._order_counter += 1
            return self._order_counter

    def clear(self) -> None:
        """Clear all stored events and reset the order counter."""
        with self._lock:
            self._events.clear()
            self._order_counter = 0
