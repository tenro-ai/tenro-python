# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Core types and context management for Tenro SDK.

Provides:
- Span types (LLMCall, ToolCall, AgentRun) for operation tracking
- SpanEvent for immutable event records
- Context functions for async-safe span stack tracking
- LifecycleManager for span lifecycle with event emission
"""

from __future__ import annotations

from tenro._core.context import (
    clear_context,
    get_current_span,
    get_span_stack,
    get_trace_id,
    pop_span,
    push_span,
)
from tenro._core.events import SpanEvent
from tenro._core.lifecycle_manager import LifecycleManager
from tenro._core.response_types import ProviderResponse
from tenro._core.spans import AgentRun, BaseSpan, LLMCall, ToolCall
from tenro._core.trace_types import SpanAttributes, TraceContext
from tenro.evals.types import EvalResult, EvalScore

__all__ = [
    "AgentRun",
    # Span types
    "BaseSpan",
    # Types
    "EvalResult",
    "EvalScore",
    "LLMCall",
    # Lifecycle
    "LifecycleManager",
    "ProviderResponse",
    "SpanAttributes",
    # Events
    "SpanEvent",
    "ToolCall",
    "TraceContext",
    "clear_context",
    "get_current_span",
    "get_span_stack",
    "get_trace_id",
    "pop_span",
    # Context
    "push_span",
]
