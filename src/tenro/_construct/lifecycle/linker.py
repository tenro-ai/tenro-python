# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Span linking for lifecycle tracking.

Provides context managers for creating and tracking spans (AgentRun, ToolCall).
LLMCall spans are created by HTTP interception, not by this module.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from tenro._core.spans import AgentRun, ToolCall
from tenro.linking.span_factories import create_agent_span, create_tool_span

if TYPE_CHECKING:
    from tenro._core.lifecycle_manager import LifecycleManager


class SpanLinker:
    """Creates and links spans to the lifecycle manager.

    Provides context managers that create spans with automatic:
    - Parent tracking (nested spans get correct parent)
    - Latency calculation (start/end times)
    - Stack-safe cleanup
    """

    def __init__(self, lifecycle: LifecycleManager) -> None:
        """Initialize with lifecycle manager.

        Args:
            lifecycle: Lifecycle manager for span tracking.
        """
        self._lifecycle = lifecycle

    @contextmanager
    def link_agent(
        self, name: str, input_data: Any = None, **kwargs: Any
    ) -> Generator[AgentRun, None, None]:
        """Link an agent execution with automatic lifecycle management.

        Args:
            name: Agent name.
            input_data: Input data for the agent.
            **kwargs: Additional keyword arguments passed to the agent.

        Yields:
            Mutable AgentRun span for direct modification.
        """
        span = create_agent_span(name, name, input_data=input_data, kwargs=kwargs)
        with self._lifecycle.start_span(span):
            yield span

    @contextmanager
    def link_tool(
        self, tool_name: str, *args: Any, **kwargs: Any
    ) -> Generator[ToolCall, None, None]:
        """Link a tool call with automatic lifecycle management.

        Args:
            tool_name: Name of the tool being called.
            *args: Positional arguments passed to the tool.
            **kwargs: Keyword arguments passed to the tool.

        Yields:
            Mutable ToolCall span for direct modification.
        """
        span = create_tool_span(tool_name, tool_name, args=args, kwargs=kwargs)
        with self._lifecycle.start_span(span):
            yield span


__all__ = ["SpanLinker"]
