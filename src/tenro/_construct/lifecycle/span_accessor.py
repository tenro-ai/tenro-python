# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Span accessor for retrieving typed spans from event store."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from tenro._core.reconstruction import (
    _create_typed_spans,
    _group_events_by_span,
    reconstruct_spans,
)
from tenro._core.spans import AgentRun, LLMCall, LLMScope, ToolCall

if TYPE_CHECKING:
    from tenro._construct.in_memory_store import EventStore

T = TypeVar("T", LLMCall, ToolCall)


class SpanAccessor:
    """Provides access to spans from event store.

    Reconstructs spans from events and provides flat lists of agent runs,
    LLM calls, and tool calls.
    """

    def __init__(self, event_store: EventStore) -> None:
        """Initialize with event store.

        Args:
            event_store: Event store containing span events.
        """
        self._event_store = event_store

    def get_root_agent_runs(self) -> list[AgentRun]:
        """Get root agent runs with populated spans.

        Returns:
            List of root agent runs with nested spans.
        """
        events = self._event_store.get_all_events()
        return reconstruct_spans(events)

    def get_all_agent_runs(self) -> list[AgentRun]:
        """Get all agent runs as flat list.

        Returns:
            Flat list including nested agents.
        """
        root_agents = self.get_root_agent_runs()
        flat_agents: list[AgentRun] = []

        def collect(agent: AgentRun) -> None:
            flat_agents.append(agent)
            for span in agent.spans:
                if isinstance(span, AgentRun):
                    collect(span)

        for root in root_agents:
            collect(root)

        return flat_agents

    def get_llm_calls(self) -> list[LLMCall]:
        """Get all LLM calls including orphans.

        Returns:
            Flat list of LLM calls from agents and orphan calls.
        """
        return self._get_spans_with_orphans(
            span_type=LLMCall,
            get_from_agent=lambda a: a.get_llm_calls(recursive=True),
        )

    def get_tool_calls(self) -> list[ToolCall]:
        """Get all tool calls including orphans.

        Returns:
            Flat list of tool calls from agents and orphan calls.
        """
        return self._get_spans_with_orphans(
            span_type=ToolCall,
            get_from_agent=lambda a: a.get_tool_calls(recursive=True),
        )

    def get_llm_scopes(self) -> list[LLMScope]:
        """Get all LLMScope spans (from @link_llm decorator).

        LLMScope tracks when @link_llm-decorated functions execute, used to
        detect missing HTTP calls.

        Returns:
            List of LLMScope spans.
        """
        events = self._event_store.get_all_events()
        span_map = _group_events_by_span(events)
        typed_spans = _create_typed_spans(span_map)
        return [span for span in typed_spans.values() if isinstance(span, LLMScope)]

    def _get_spans_with_orphans(
        self,
        span_type: type[T],
        get_from_agent: Callable[[AgentRun], list[T]],
    ) -> list[T]:
        """Get spans from agents plus orphan spans.

        Args:
            span_type: Type of span to retrieve.
            get_from_agent: Function to get spans from an agent.

        Returns:
            Combined list of agent spans and orphan spans.
        """
        root_agents = self.get_root_agent_runs()
        agent_spans = [span for agent in root_agents for span in get_from_agent(agent)]
        agent_span_ids = {span.id for span in agent_spans}

        events = self._event_store.get_all_events()
        span_map = _group_events_by_span(events)
        typed_spans = _create_typed_spans(span_map)

        orphans = [
            span
            for span in typed_spans.values()
            if isinstance(span, span_type) and span.id not in agent_span_ids
        ]

        return agent_spans + orphans


__all__ = ["SpanAccessor"]
