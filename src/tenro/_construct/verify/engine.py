# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Core verification engine for tool, agent, and LLM verifications.

Provides shared verification logic across tool, agent, and LLM events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from tenro._construct.verify.assertions import CallCountAssertion
from tenro._core.span_filters import filter_by_agent
from tenro.errors import TenroVerificationError

if TYPE_CHECKING:
    from tenro._core.spans import AgentRun, LLMCall, ToolCall

EventType = Literal["tool", "agent", "llm"]


def _match_arguments(
    call: ToolCall | AgentRun,
    called_with: dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> bool:
    """Check if a call's arguments match the expected values.

    Args:
        call: The tool or agent call to check.
        called_with: Dict of expected arguments. Use when a key conflicts with
            a verification parameter like `times`.
        kwargs: Expected keyword arguments.

    Returns:
        `True` if all expected arguments match, `False` otherwise.
    """
    expected_kwargs = called_with if called_with is not None else kwargs

    if not hasattr(call, "kwargs"):
        return False

    for key, expected_value in expected_kwargs.items():
        actual_value = call.kwargs.get(key)
        if actual_value != expected_value:
            return False

    return True


def _filter_by_name(
    calls: Any,
    name: str | tuple[str, ...] | None,
    event_type: EventType,
) -> Any:
    """Filter calls by name (tool_name/target_path or agent name/target_path).

    Args:
        calls: List of calls to filter.
        name: Name(s) to filter by. Can be a single string or tuple of paths.
            For class targets with multiple entry methods, a tuple allows
            matching ANY of the entry method paths.
        event_type: Type of event being filtered.

    Returns:
        Filtered list of calls.
    """
    if not name:
        return calls

    names = (name,) if isinstance(name, str) else name

    # display_name is for trace output only - matching uses target_path
    path_names = set(names)

    if event_type == "tool":
        return [c for c in calls if hasattr(c, "target_path") and c.target_path in path_names]
    elif event_type == "agent":
        return [c for c in calls if c.target_path in path_names]
    else:  # llm - filter by target_path when specified
        return [c for c in calls if c.target_path in path_names]


def _filter_by_provider(calls: list[LLMCall], provider: str | None) -> list[LLMCall]:
    """Filter LLM calls by provider.

    Args:
        calls: List of LLM calls to filter.
        provider: Provider name to filter by (e.g., "openai", "anthropic").

    Returns:
        Filtered list of LLM calls.
    """
    if not provider:
        return calls
    return [c for c in calls if c.provider == provider]


def verify_call_count(
    calls: Any,
    agent_runs: list[AgentRun],
    count: int | None,
    min: int | None,
    max: int | None,
    name_filter: str | tuple[str, ...] | None,
    agent_filter: str | None,
    event_type: EventType,
    provider_filter: str | None = None,
) -> None:
    """Verify call count with optional filters.

    This is the core verification function used by all verify_*() methods.

    Args:
        calls: List of calls to verify (ToolCall, AgentRun, or LLMCall).
        agent_runs: List of agent runs for filtering.
        count: Expected exact count (mutually exclusive with min/max).
        min: Minimum count (inclusive).
        max: Maximum count (inclusive).
        name_filter: Optional name filter (tool name or agent name). Can be a
            tuple of paths to match ANY of multiple entry method paths.
        agent_filter: Optional agent name filter.
        event_type: Type of event ("tool", "agent", "llm").
        provider_filter: Optional provider filter (LLM only).

    Raises:
        AssertionError: If count doesn't match expected criteria.
        ValueError: If count and min/max are both specified.
    """
    assertion = CallCountAssertion(count=count, min=min, max=max)
    filtered = _filter_by_name(calls, name_filter, event_type)

    if event_type == "llm" and provider_filter:
        filtered = _filter_by_provider(filtered, provider_filter)

    filtered = filter_by_agent(agent_filter, filtered, agent_runs)
    actual_count = len(filtered)
    if not assertion.matches(actual_count):
        context_parts = []
        if name_filter:
            display_name = name_filter[0] if isinstance(name_filter, tuple) else name_filter
            if event_type == "tool":
                context_parts.append(f"tool '{display_name}'")
            elif event_type == "agent":
                context_parts.append(f"agent '{display_name}'")
        if provider_filter:
            context_parts.append(f"provider='{provider_filter}'")
        if agent_filter:
            context_parts.append(f"by agent '{agent_filter}'")

        context = f" for {' '.join(context_parts)}" if context_parts else ""
        raise TenroVerificationError(assertion.error_message(actual_count, context))


def verify_with_arguments(
    calls: Any,
    agent_runs: list[AgentRun],
    name: str | tuple[str, ...],
    called_with: dict[str, Any] | None,
    agent_filter: str | None,
    event_type: EventType,
    kwargs: dict[str, Any],
) -> None:
    """Verify calls with argument matching (at least once).

    Args:
        calls: List of calls to verify (ToolCall or AgentRun).
        agent_runs: List of agent runs for filtering.
        name: Name of tool or agent. Can be a tuple of paths to match ANY of
            multiple entry method paths.
        called_with: Dict of expected arguments.
        agent_filter: Optional agent name filter.
        event_type: Type of event ("tool" or "agent").
        kwargs: Expected keyword arguments.

    Raises:
        AssertionError: If no matching calls found.
    """
    filtered = _filter_by_name(calls, name, event_type)
    filtered = filter_by_agent(agent_filter, filtered, agent_runs)
    matching_calls = [c for c in filtered if _match_arguments(c, called_with, kwargs)]
    actual_count = len(matching_calls)

    display_name = name[0] if isinstance(name, tuple) else name

    if actual_count == 0:
        expected_args = called_with if called_with else kwargs
        event_name = "tool" if event_type == "tool" else "agent"

        if filtered:
            actual_args_list = [getattr(c, "kwargs", {}) for c in filtered if hasattr(c, "kwargs")]
            if actual_args_list:
                actual_args_str = ", ".join(str(a) for a in actual_args_list[:3])
                if len(actual_args_list) > 3:
                    actual_args_str += f", ... ({len(actual_args_list)} total)"
                hint = f"\n\nRecorded {event_name} arguments: {actual_args_str}"
            else:
                hint = f"\n\nNo arguments were recorded for this {event_name}."
        else:
            hint = f"\n\nNo calls to '{display_name}' were recorded."

        raise TenroVerificationError(
            f"Expected {event_name} '{display_name}' to be called at least once with "
            f"arguments {expected_args}, but found no matching calls.{hint}"
        )


def verify_sequence(
    calls: Any,
    expected_sequence: list[str],
    event_type: EventType,
) -> None:
    """Verify calls were made in a specific order.

    Args:
        calls: List of calls to verify (ToolCall or AgentRun).
        expected_sequence: Expected sequence of names.
        event_type: Type of event ("tool" or "agent").

    Raises:
        AssertionError: If sequence doesn't match.
    """
    if event_type == "tool":
        actual_sequence = [c.display_name for c in calls if hasattr(c, "display_name")]
        event_name = "tool"
    elif event_type == "agent":
        actual_sequence = [c.display_name for c in calls]
        event_name = "agent"
    else:
        raise ValueError(f"Sequence verification not supported for event type: {event_type}")

    if actual_sequence != expected_sequence:
        raise TenroVerificationError(
            f"Expected {event_name} sequence {expected_sequence}, but got {actual_sequence}."
        )


__all__ = [
    "verify_call_count",
    "verify_sequence",
    "verify_with_arguments",
]
