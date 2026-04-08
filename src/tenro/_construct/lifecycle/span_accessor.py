# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Span accessor for retrieving typed spans from span store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tenro._core.spans import AgentRun, LLMCall, LLMScope, ToolCall

if TYPE_CHECKING:
    from tenro._construct.span_store import SpanStore


class SpanAccessor:
    """Provides typed access to spans from span store."""

    def __init__(self, span_store: SpanStore) -> None:
        """Initialize with span store.

        Args:
            span_store: Store containing completed spans.
        """
        self._span_store = span_store

    def get_root_agent_runs(self) -> list[AgentRun]:
        """Get root agent runs with populated child spans.

        Root agents are those with no parent_agent_id. Their .spans
        list is already populated by LifecycleManager during execution.

        Returns:
            List of root agent runs sorted by start time.
        """
        spans = self._span_store.get_all_spans()
        return sorted(
            (s for s in spans if isinstance(s, AgentRun) and s.parent_agent_id is None),
            key=lambda s: (s.start_time, s.creation_seq),
        )

    def get_all_agent_runs(self) -> list[AgentRun]:
        """Get all agent runs as flat list.

        Returns:
            Flat list including nested agents, sorted by start time.
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
            Flat list of all LLM calls sorted by start time.
        """
        spans = self._span_store.get_all_spans()
        return sorted(
            (s for s in spans if isinstance(s, LLMCall)),
            key=lambda s: (s.start_time, s.creation_seq),
        )

    def get_tool_calls(self) -> list[ToolCall]:
        """Get all tool calls including orphans.

        Returns:
            Flat list of all tool calls sorted by start time.
        """
        spans = self._span_store.get_all_spans()
        return sorted(
            (s for s in spans if isinstance(s, ToolCall)),
            key=lambda s: (s.start_time, s.creation_seq),
        )

    def get_llm_scopes(self) -> list[LLMScope]:
        """Get all LLMScope spans (from @link_llm decorator).

        Returns:
            List of LLMScope spans sorted by start time.
        """
        spans = self._span_store.get_all_spans()
        return sorted(
            (s for s in spans if isinstance(s, LLMScope)),
            key=lambda s: (s.start_time, s.creation_seq),
        )


__all__ = ["SpanAccessor"]
