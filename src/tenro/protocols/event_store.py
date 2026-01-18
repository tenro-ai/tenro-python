# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Protocol definitions for Tenro SDK interfaces.

Defines Protocols for storage backends used by the SDK.
"""

from __future__ import annotations

from typing import Protocol

from tenro._core.events import SpanEvent


class EventStore(Protocol):
    """Protocol for event storage backends.

    Any class that implements these methods can be used as an event store.
    Common implementations include in-memory stores, file-backed stores, or
    remote stores for distributed tracing.

    Examples:
        >>> class MyEventStore:
        ...     def emit(self, event: SpanEvent) -> None: ...
        ...     def get_all_events(self) -> list[SpanEvent]: ...
        ...     def get_next_order_index(self) -> int: ...
        ...     def clear(self) -> None: ...
        >>>
        >>> # MyEventStore is compatible with EventStore protocol
    """

    def emit(self, event: SpanEvent) -> None:
        """Store an event.

        Args:
            event: Event to persist.
        """
        ...

    def get_all_events(self) -> list[SpanEvent]:
        """Retrieve all stored events.

        Returns:
            All events sorted by (timestamp, order_index).
        """
        ...

    def get_next_order_index(self) -> int:
        """Get next sequential order index.

        Returns:
            Incrementing integer for order_index field.
        """
        ...

    def clear(self) -> None:
        """Clear all events."""
        ...
