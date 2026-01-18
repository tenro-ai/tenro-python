# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Tool simulation API.

- tool.verify(target)           # at least once (1+)
- tool.verify_many(count=1)     # exactly once
- tool.verify_many(count=2)     # exactly twice
- tool.verify_never(target)     # never called
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tenro.simulate._helpers import get_construct_or_raise

_UNSET: Any = object()

if TYPE_CHECKING:
    from collections.abc import Callable

    from tenro.spans import ToolCallSpan


def simulate(
    target: str | Callable[..., Any],
    result: Any = None,
    results: list[Any] | None = None,
    side_effect: Callable[..., Any] | None = None,
    optional: bool = False,
) -> None:
    """Register a tool simulation."""
    get_construct_or_raise().simulate_tool(
        target, result=result, results=results, side_effect=side_effect, optional=optional
    )


def verify(
    target: str | Callable[..., Any],
    *,
    result: Any = _UNSET,
    where: Callable[[ToolCallSpan], bool] | None = None,
    **kwargs: Any,
) -> ToolCallSpan:
    """Verify a tool was called at least once (1+).

    Use verify_many(count=N) for exact count verification.

    Args:
        target: Tool function or dotted path to verify.
        result: Expected return value (exact equality check). Use to verify
            the tool returned a specific value including None.
        where: Predicate filter for span selection.
        **kwargs: Additional arguments for matching tool arguments.

    Returns:
        The matching tool call span.
    """
    verify_kwargs: dict[str, Any] = {"where": where}
    if result is not _UNSET:
        verify_kwargs["result"] = result
    span: ToolCallSpan = get_construct_or_raise().verify_tool(target, **verify_kwargs, **kwargs)
    return span


def verify_never(target: str | Callable[..., Any]) -> None:
    """Verify a tool was never called."""
    get_construct_or_raise().verify_tool_never(target)


def verify_many(
    target: str | Callable[..., Any] | None = None,
    *,
    where: Callable[[ToolCallSpan], bool] | None = None,
    count: int | None = None,
    at_least: int | None = None,
    at_most: int | None = None,
) -> tuple[ToolCallSpan, ...]:
    """Verify tool call count.

    Args:
        target: Tool function or dotted path to filter by.
        where: Predicate filter for span selection.
        count: Exact number of calls expected.
        at_least: Minimum number of calls expected.
        at_most: Maximum number of calls expected.

    Examples:
        >>> tool.verify_many(search, count=1)      # exactly once
        >>> tool.verify_many(search, count=2)      # exactly twice
        >>> tool.verify_many(search, at_least=1)   # 1 or more
    """
    resolved_target: str | tuple[str, ...] | None = None
    if callable(target):
        from tenro._construct.simulate.target_resolution import (
            resolve_all_target_paths,
        )

        resolved = resolve_all_target_paths(target, is_tool=True)
        # Pass ALL paths for multi-entry tools
        resolved_target = (
            tuple(resolved) if len(resolved) > 1 else (resolved[0] if resolved else None)
        )
    else:
        resolved_target = target
    get_construct_or_raise().verify_tools(
        count=count, min=at_least, max=at_most, target=resolved_target
    )
    all_calls = calls()
    if where:
        return tuple(c for c in all_calls if where(c))
    return all_calls


def verify_sequence(
    expected_sequence: list[str | Callable[..., Any]],
) -> tuple[ToolCallSpan, ...]:
    """Verify tools were called in a specific order."""
    sequence = [t.__name__ if callable(t) else t for t in expected_sequence]
    get_construct_or_raise().verify_tool_sequence(sequence)
    return calls()[: len(expected_sequence)]


def calls(
    target: str | Callable[..., Any] | None = None,
) -> tuple[ToolCallSpan, ...]:
    """Get tool calls, optionally filtered by target.

    Args:
        target: Tool function or dotted path to filter by. If None, returns all calls.

    Returns:
        Tuple of matching tool call spans.
    """
    all_calls = tuple(get_construct_or_raise().tool_calls)
    if target is None:
        return all_calls

    if callable(target):
        from tenro._construct.simulate.target_resolution import resolve_all_target_paths

        paths = resolve_all_target_paths(target, is_tool=True)
    else:
        paths = (target,)

    return tuple(c for c in all_calls if c.target_path in paths)


__all__ = ["calls", "simulate", "verify", "verify_many", "verify_never", "verify_sequence"]
