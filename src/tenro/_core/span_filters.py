# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Utilities for filtering span objects by various criteria.

This module provides generic filtering functions that work with all span types
(ToolCall, LLMCall, AgentRun), which all inherit from BaseSpan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from tenro._core.spans import AgentRun, BaseSpan

T = TypeVar("T", bound="BaseSpan")


def filter_by_agent(agent: str | None, spans: list[T], agent_runs: list[AgentRun]) -> list[T]:
    """Filter spans by agent name using agent_id relationship.

    Works with any span type (ToolCall, LLMCall, AgentRun):
    - ToolCall/LLMCall: Filters by agent_id (the agent that made the call)
    - AgentRun: Filters by parent_agent_id (child agents spawned by parent)

    Args:
        agent: Agent name to filter by. If `None`, returns all spans.
        spans: List of spans to filter (generic type T).
        agent_runs: List of agent runs to search for matching agent.

    Returns:
        Filtered list of spans created by the specified agent.
        Returns empty list if agent not found.
        Returns all spans if agent is `None`.

    Examples:
        >>> # Filter tool calls by agent
        >>> manager_tools = filter_by_agent("Manager", tool_calls, agent_runs)
        >>> # Filter LLM calls by agent
        >>> researcher_llms = filter_by_agent("Researcher", llm_calls, agent_runs)
        >>> # Filter child agents by parent
        >>> manager_children = filter_by_agent("Manager", agent_runs, agent_runs)
        >>> # No filter - returns all
        >>> all_spans = filter_by_agent(None, spans, agent_runs)
    """
    if not agent:
        return spans

    from tenro._core.spans import AgentRun as AgentRunClass

    matching_agents = [a for a in agent_runs if a.display_name == agent or a.target_path == agent]
    if not matching_agents:
        return []
    matching_ids = {a.span_id for a in matching_agents}

    result: list[T] = []
    for span in spans:
        if isinstance(span, AgentRunClass):
            if span.parent_agent_id in matching_ids:
                result.append(span)
        elif span.agent_id in matching_ids:
            result.append(span)
    return result


__all__ = ["filter_by_agent"]
