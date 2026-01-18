# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Span reconstruction algorithm.

Converts flat event list into a hierarchical span tree.
"""

from __future__ import annotations

from typing import Any

from tenro._core.events import SpanEvent
from tenro._core.spans import AgentRun, BaseSpan, LLMCall, LLMScope, ToolCall


def reconstruct_spans(
    events: list[SpanEvent],
    max_safe_depth: int = 50,
) -> list[AgentRun]:
    """Reconstruct hierarchical spans from flat event list.

    Steps:
    1. Group events by span_id.
    2. Convert to typed span objects.
    3. Build hierarchy using parent_span_id.

    Args:
        events: Flat list of span events.
        max_safe_depth: Maximum tree depth (safety limit).

    Returns:
        Root agent runs with populated .spans field.
    """
    if not events:
        return []

    span_map = _group_events_by_span(events)
    typed_spans = _create_typed_spans(span_map)
    root_spans = _build_hierarchy(typed_spans, max_safe_depth)

    return root_spans


def _group_events_by_span(events: list[SpanEvent]) -> dict[str, dict[str, Any]]:
    """Group events by span_id and merge into span data."""
    span_map: dict[str, dict[str, Any]] = {}

    for event in sorted(events, key=lambda e: (e.timestamp, e.order_index)):
        span_id = event.span_id

        if span_id not in span_map:
            span_map[span_id] = {
                "id": span_id,
                "trace_id": event.trace_id,
                "parent_span_id": event.parent_span_id,
                "span_kind": event.span_kind,
                "start_time": event.timestamp,
                "status": "running",
            }

        if event.event_type == "start":
            span_map[span_id].update(event.data)
            span_map[span_id]["start_time"] = event.timestamp
        elif event.event_type == "end":
            span_map[span_id]["status"] = "completed"
            span_map[span_id]["end_time"] = event.timestamp
            span_map[span_id].update(event.data)
        elif event.event_type == "error":
            span_map[span_id]["status"] = "error"
            span_map[span_id]["error"] = event.data.get("message")

    return span_map


def _create_typed_spans(span_map: dict[str, dict[str, Any]]) -> dict[str, BaseSpan]:
    """Convert span data to typed span objects."""
    typed_spans: dict[str, BaseSpan] = {}

    for span_id, span_data in span_map.items():
        span_kind = span_data.pop("span_kind")
        parent_id = span_data.pop("parent_span_id")

        _calculate_latency(span_data)
        span = _create_span_by_kind(span_kind, span_data, parent_id)
        if span:
            typed_spans[span_id] = span

    return typed_spans


def _calculate_latency(span_data: dict[str, Any]) -> None:
    """Calculate and add latency_ms if end time exists."""
    if "end_time" in span_data:
        start = span_data.get("start_time", 0)
        end = span_data.pop("end_time")
        span_data["latency_ms"] = int((end - start) * 1000)


def _create_span_by_kind(
    span_kind: str, span_data: dict[str, Any], parent_id: str | None
) -> BaseSpan | None:
    """Create typed span based on kind.

    Sets `parent_span_id` to immediate parent (for LLM↔Tool linking).
    `agent_id` is set later in `_build_hierarchy` by walking ancestors.
    """
    span: BaseSpan
    if span_kind == "AGENT":
        agent_name = span_data.pop("agent_name", None)
        # Map agent_name to display_name and ensure target_path is set
        if "display_name" not in span_data:
            span_data["display_name"] = agent_name
        if "target_path" not in span_data:
            span_data["target_path"] = agent_name if agent_name is not None else span_data["id"]
        span = AgentRun(**span_data, spans=[])
        span.parent_span_id = parent_id
        span.parent_agent_id = parent_id  # For agents, parent is always an agent
        return span
    elif span_kind == "LLM":
        _map_llm_fields(span_data)
        span = LLMCall(**span_data)
        span.parent_span_id = parent_id  # Immediate parent (could be Agent, LLM, or Tool)
        return span
    elif span_kind == "TOOL":
        # Map tool_name to display_name and ensure target_path is set
        tool_name = span_data.pop("tool_name", None)
        if "display_name" not in span_data:
            span_data["display_name"] = tool_name
        if "target_path" not in span_data:
            span_data["target_path"] = tool_name if tool_name is not None else span_data["id"]
        span = ToolCall(**span_data)
        span.parent_span_id = parent_id  # Immediate parent (could be Agent, LLM, or Tool)
        return span
    elif span_kind == "LLM_SCOPE":
        span = LLMScope(**span_data)
        span.parent_span_id = parent_id
        return span
    return None


def _map_llm_fields(span_data: dict[str, Any]) -> None:
    """Map event payload fields to span fields."""
    if "text" in span_data:
        span_data["response"] = span_data.pop("text")
    if "usage" in span_data:
        span_data["token_usage"] = span_data.pop("usage")


def _build_hierarchy(
    typed_spans: dict[str, BaseSpan],
    max_safe_depth: int,
) -> list[AgentRun]:
    """Build parent-child hierarchy from flat span list.

    For each span:
    1. Walk parent chain to find closest AgentRun ancestor
    2. Set `agent_id` to that agent's ID
    3. Add as child to that agent's `.spans` list
    """
    root_spans: list[AgentRun] = []

    for span_id, span in typed_spans.items():
        # Find the closest agent ancestor by walking parent_span_id chain
        agent_ancestor = _find_agent_ancestor(span, typed_spans)

        if agent_ancestor is None:
            if isinstance(span, AgentRun):
                span.agent_id = span.id
                root_spans.append(span)
            continue

        if isinstance(span, AgentRun):
            span.agent_id = span.id
            span.parent_agent_id = agent_ancestor.id
        else:
            span.agent_id = agent_ancestor.id

        depth = _calculate_depth(span_id, typed_spans)
        if depth <= max_safe_depth:
            agent_ancestor.spans.append(span)

    return root_spans


def _find_agent_ancestor(span: BaseSpan, typed_spans: dict[str, BaseSpan]) -> AgentRun | None:
    """Walk parent chain to find closest AgentRun."""
    parent_id = span.parent_span_id
    visited: set[str] = {span.id}

    while parent_id and parent_id not in visited:
        visited.add(parent_id)
        parent = typed_spans.get(parent_id)
        if parent is None:
            return None
        if isinstance(parent, AgentRun):
            return parent
        parent_id = parent.parent_span_id

    return None


def _calculate_depth(
    span_id: str,
    spans: dict[str, BaseSpan],
    depth: int = 0,
    visited: set[str] | None = None,
) -> int:
    """Calculate depth of span in hierarchy using parent_span_id.

    Includes cycle detection to prevent infinite recursion.
    """
    if visited is None:
        visited = set()

    if span_id in visited:
        return depth  # Cycle detected, stop recursion

    visited.add(span_id)

    span = spans.get(span_id)
    if not span:
        return depth

    parent_id = span.parent_span_id
    if parent_id is None:
        return depth

    return _calculate_depth(parent_id, spans, depth + 1, visited)


__all__ = ["_create_typed_spans", "_group_events_by_span", "reconstruct_spans"]
