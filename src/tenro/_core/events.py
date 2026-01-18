# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Event models for span tracking.

Immutable events are used to reconstruct spans on read.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from uuid_utils import uuid7

from tenro._core.model_base import BaseModel


class SpanEvent(BaseModel):
    """Immutable event representing a span lifecycle event.

    Attributes:
        event_id: Unique event identifier.
        trace_id: Trace identifier for the overall workflow.
        span_id: Span identifier for a single operation.
        parent_span_id: Parent span identifier when nested.
        event_type: Lifecycle event type (`start`, `end`, or `error`).
        span_kind: Span category (`AGENT`, `LLM`, or `TOOL`).
        timestamp: Unix timestamp when the event was emitted.
        order_index: Monotonic order within the span stream.
        data: Payload data for the event.
    """

    event_id: str
    trace_id: str
    span_id: str
    parent_span_id: str | None

    event_type: Literal["start", "end", "error"]
    span_kind: Literal["AGENT", "LLM", "TOOL", "LLM_SCOPE"]
    timestamp: float
    order_index: int

    data: dict[str, Any]

    @classmethod
    def create_start(
        cls,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        span_kind: Literal["AGENT", "LLM", "TOOL", "LLM_SCOPE"],
        data: dict[str, Any],
        order_index: int = 0,
    ) -> SpanEvent:
        """Create a start event."""
        return cls(
            event_id=str(uuid7()),
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_type="start",
            span_kind=span_kind,
            timestamp=time.time(),
            order_index=order_index,
            data=data,
        )

    @classmethod
    def create_end(
        cls,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        span_kind: Literal["AGENT", "LLM", "TOOL", "LLM_SCOPE"],
        data: dict[str, Any],
        order_index: int = 0,
    ) -> SpanEvent:
        """Create an end event."""
        return cls(
            event_id=str(uuid7()),
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_type="end",
            span_kind=span_kind,
            timestamp=time.time(),
            order_index=order_index,
            data=data,
        )

    @classmethod
    def create_error(
        cls,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        span_kind: Literal["AGENT", "LLM", "TOOL", "LLM_SCOPE"],
        error_message: str,
        order_index: int = 0,
    ) -> SpanEvent:
        """Create an error event."""
        return cls(
            event_id=str(uuid7()),
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_type="error",
            span_kind=span_kind,
            timestamp=time.time(),
            order_index=order_index,
            data={"message": error_message},
        )
