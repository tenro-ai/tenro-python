# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Agent simulation API.

- agent.verify(target)          # at least once (1+)
- agent.verify_many(count=1)    # exactly once
- agent.verify_many(count=2)    # exactly twice
- agent.verify_never(target)    # never called
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tenro.simulate._helpers import get_construct_or_raise

_UNSET: Any = object()

if TYPE_CHECKING:
    from collections.abc import Callable

    from tenro.spans import AgentRunSpan


def simulate(
    target: str | Callable[..., Any],
    result: Any = None,
    results: list[Any] | None = None,
    side_effect: Callable[..., Any] | None = None,
    optional: bool = False,
) -> None:
    """Register an agent simulation."""
    get_construct_or_raise().simulate_agent(
        target, result=result, results=results, side_effect=side_effect, optional=optional
    )


def verify(
    target: str | Callable[..., Any],
    *,
    result: Any = _UNSET,
    where: Callable[[AgentRunSpan], bool] | None = None,
) -> AgentRunSpan:
    """Verify an agent was called at least once (1+).

    Use verify_many(count=N) for exact count verification.

    Args:
        target: Agent function or dotted path to verify.
        result: Expected return value (exact equality check). Use to verify
            the agent returned a specific value including None.
        where: Predicate filter for span selection.

    Returns:
        The matching agent run span.
    """
    verify_kwargs: dict[str, Any] = {"where": where}
    if result is not _UNSET:
        verify_kwargs["result"] = result
    span: AgentRunSpan = get_construct_or_raise().verify_agent(target, **verify_kwargs)
    return span


def verify_never(target: str | Callable[..., Any]) -> None:
    """Verify an agent was never called."""
    get_construct_or_raise().verify_agent_never(target)


def verify_many(
    target: str | Callable[..., Any] | None = None,
    *,
    where: Callable[[AgentRunSpan], bool] | None = None,
    count: int | None = None,
    at_least: int | None = None,
    at_most: int | None = None,
) -> tuple[AgentRunSpan, ...]:
    """Verify agent call count.

    Args:
        target: Agent function or dotted path to filter by.
        where: Predicate filter for span selection.
        count: Exact number of calls expected.
        at_least: Minimum number of calls expected.
        at_most: Maximum number of calls expected.

    Examples:
        >>> agent.verify_many(MyAgent, count=1)      # exactly once
        >>> agent.verify_many(MyAgent, count=2)      # exactly twice
        >>> agent.verify_many(MyAgent, at_least=1)   # 1 or more
    """
    if callable(target):
        from tenro._construct.simulate.target_resolution import (
            resolve_all_target_paths,
        )

        resolved = resolve_all_target_paths(target)
        resolved_target: str | tuple[str, ...] | None = (
            resolved[0] if len(resolved) == 1 else resolved
        )
    else:
        resolved_target = target
    get_construct_or_raise().verify_agents(
        count=count, min=at_least, max=at_most, target=resolved_target
    )
    all_runs = runs()
    if where:
        return tuple(r for r in all_runs if where(r))
    return all_runs


def verify_sequence(
    expected_sequence: list[str | Callable[..., Any]],
) -> tuple[AgentRunSpan, ...]:
    """Verify agents were called in a specific order."""
    sequence = [a.__name__ if callable(a) else a for a in expected_sequence]
    get_construct_or_raise().verify_agent_sequence(sequence)
    return runs()[: len(expected_sequence)]


def runs() -> tuple[AgentRunSpan, ...]:
    """Get all agent runs (read-only access)."""
    return tuple(get_construct_or_raise().agent_runs)


__all__ = ["runs", "simulate", "verify", "verify_many", "verify_never", "verify_sequence"]
