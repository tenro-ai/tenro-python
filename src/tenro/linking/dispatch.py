# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Dispatch helpers for call-time simulation resolution.

These helpers check for simulation rules at call time and dispatch to either
the simulated behavior or the real implementation. This enables captured
references to work correctly regardless of when simulate() was called.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import TYPE_CHECKING, Any, TypeVar

from tenro.errors.simulation import TenroSimulationExecutionError
from tenro.linking.context import get_construct_stack

if TYPE_CHECKING:
    from tenro._construct.construct import Construct
    from tenro._construct.simulate.rule import SimulationRule

T = TypeVar("T")


def _get_rule_from_stack(canonical_key: str) -> tuple[Construct, SimulationRule] | None:
    """Search the construct stack for a simulation rule.

    Searches from innermost (top) to outermost (bottom) construct.
    This enables nested constructs where inner can override outer's simulations.

    Args:
        canonical_key: The target's canonical dotted path.

    Returns:
        Tuple of (construct_that_has_rule, rule) if found, None otherwise.
    """
    stack = get_construct_stack()
    for construct in reversed(stack):
        rule = construct._orchestrator._simulations.get(canonical_key)
        if rule is not None:
            return construct, rule
    return None


def get_simulation_model(canonical_key: str, default: str | None = None) -> str | None:
    """Get the model from a simulation rule, or a default.

    Used by @link_llm decorator to get the model specified in simulate_llm()
    when creating LLMCall spans.

    Args:
        canonical_key: The target's canonical dotted path.
        default: Fallback value if no simulation model exists.

    Returns:
        The simulation model, or default if none.
    """
    found = _get_rule_from_stack(canonical_key)
    if found is not None and found[1].llm_model is not None:
        return found[1].llm_model
    return default


def _mark_triggered(
    construct: Construct,
    canonical_key: str,
    rule: SimulationRule | None = None,
) -> None:
    """Mark a simulation target as triggered in the appropriate tracker.

    Args:
        construct: The active Construct instance.
        canonical_key: The target's canonical dotted path.
        rule: Optional simulation rule (provides llm_provider for LLM simulations).
    """
    orchestrator = construct._orchestrator
    if canonical_key in orchestrator._tool_tracker._registered:
        orchestrator._tool_tracker.mark_triggered(canonical_key)
    elif canonical_key in orchestrator._agent_tracker._registered:
        orchestrator._agent_tracker.mark_triggered(canonical_key)
    elif rule is not None and hasattr(rule, "llm_provider") and rule.llm_provider:
        # LLM simulations track by provider, not target path
        orchestrator._simulation_tracker.mark_triggered(rule.llm_provider)


def _execute_rule(
    rule: SimulationRule,
    canonical_key: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Execute a simulation rule and return the result.

    Args:
        rule: The simulation rule.
        canonical_key: For error messages.
        args: Call arguments.
        kwargs: Call keyword arguments.

    Returns:
        The simulated result.

    Raises:
        RuntimeError: If result sequence is exhausted.
        BaseException: If result is an exception instance (raises it).
    """
    result: object = None

    if rule.side_effect is not None:
        result = rule.side_effect(*args, **kwargs)  # type: ignore[operator]
    elif rule.returns_value is not None:
        result = rule.returns_value
    elif rule.result_sequence is not None:
        if not rule.result_sequence:
            raise RuntimeError(f"Simulation sequence exhausted for {canonical_key}")
        result = rule.result_sequence.pop(0)

    if isinstance(result, BaseException):
        raise result

    return result


def dispatch_sync(
    canonical_key: str,
    call_real: Callable[..., T],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[T, bool]:
    """Dispatch for sync functions - returns (result, simulated).

    Args:
        canonical_key: The target's canonical dotted path.
        call_real: Callable to invoke the real implementation.
        args: Positional arguments for the call.
        kwargs: Keyword arguments for the call.

    Returns:
        Tuple of (result, was_simulated).

    Raises:
        TenroSimulationExecutionError: If simulation returns wrong kind.
    """
    found = _get_rule_from_stack(canonical_key)
    if found is None:
        return call_real(*args, **kwargs), False

    construct, rule = found
    _mark_triggered(construct, canonical_key, rule)
    result = _execute_rule(rule, canonical_key, args, kwargs)

    # Sync wrapper must reject awaitables (not execute them)
    # Close coroutine to prevent RuntimeWarning about unawaited coroutines
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TenroSimulationExecutionError(
            canonical_key, expected_kind="sync", actual_kind="awaitable"
        )

    if inspect.isgenerator(result) or inspect.isasyncgen(result):
        raise TenroSimulationExecutionError(
            canonical_key, expected_kind="sync", actual_kind="generator"
        )

    return result, True


async def dispatch_async(
    canonical_key: str,
    call_real_async: Callable[..., Awaitable[T]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[T, bool]:
    """Dispatch for async functions - returns (result, simulated).

    Args:
        canonical_key: The target's canonical dotted path.
        call_real_async: Async callable to invoke the real implementation.
        args: Positional arguments for the call.
        kwargs: Keyword arguments for the call.

    Returns:
        Tuple of (result, was_simulated).

    Raises:
        TenroSimulationExecutionError: If simulation returns wrong kind.
    """
    found = _get_rule_from_stack(canonical_key)
    if found is None:
        return await call_real_async(*args, **kwargs), False

    construct, rule = found
    _mark_triggered(construct, canonical_key, rule)
    result = _execute_rule(rule, canonical_key, args, kwargs)

    if inspect.isawaitable(result):
        result = await result

    if inspect.isgenerator(result) or inspect.isasyncgen(result):
        raise TenroSimulationExecutionError(
            canonical_key, expected_kind="async", actual_kind="generator"
        )

    return result, True


def _get_generator_items_static(
    rule: SimulationRule,
    canonical_key: str,
) -> list[Any] | None:
    """Get items to yield for non-side_effect generator rules.

    Args:
        rule: The simulation rule.
        canonical_key: For error messages.

    Returns:
        List of items to yield, or None if rule uses side_effect.
    """
    if rule.generator_items is not None:
        return list(rule.generator_items)

    if rule.side_effect is not None:
        return None

    if rule.returns_value is not None:
        return [rule.returns_value]

    if rule.result_sequence is not None:
        if not rule.result_sequence:
            raise RuntimeError(f"Simulation sequence exhausted for {canonical_key}")
        return [rule.result_sequence.pop(0)]

    return []


def dispatch_gen(
    canonical_key: str,
    call_real_gen: Callable[..., Generator[T, None, None]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Generator[T, None, None], bool]:
    """Dispatch for generator functions - returns (generator, simulated).

    Args:
        canonical_key: The target's canonical dotted path.
        call_real_gen: Callable returning a generator.
        args: Positional arguments for the call.
        kwargs: Keyword arguments for the call.

    Returns:
        Tuple of (generator, was_simulated).

    Raises:
        TenroSimulationExecutionError: If simulation returns wrong kind.
    """
    found = _get_rule_from_stack(canonical_key)
    if found is None:
        return call_real_gen(*args, **kwargs), False

    construct, rule = found
    _mark_triggered(construct, canonical_key, rule)

    items = _get_generator_items_static(rule, canonical_key)
    if items is not None:
        return _iter_to_gen(iter(items)), True

    result = rule.side_effect(*args, **kwargs)  # type: ignore[operator]
    return _ensure_generator(result, canonical_key), True


def _ensure_generator(result: Any, canonical_key: str) -> Generator[Any, None, None]:
    """Validate result is a Generator or raise error.

    Returns:
        The generator if valid.
    """
    if inspect.isgenerator(result):
        return result
    raise TenroSimulationExecutionError(
        canonical_key, expected_kind="generator", actual_kind=type(result).__name__
    )


def _iter_to_gen(it: Any) -> Generator[Any, None, None]:
    """Convert an iterator to a generator."""
    yield from it


async def dispatch_asyncgen(
    canonical_key: str,
    call_real_asyncgen: Callable[..., AsyncGenerator[T, None]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[AsyncGenerator[T, None], bool]:
    """Dispatch for async generator functions.

    Args:
        canonical_key: The target's canonical dotted path.
        call_real_asyncgen: Callable returning an async generator.
        args: Positional arguments for the call.
        kwargs: Keyword arguments for the call.

    Returns:
        Tuple of (async_generator, was_simulated).

    Raises:
        TenroSimulationExecutionError: If simulation returns wrong kind.
    """
    found = _get_rule_from_stack(canonical_key)
    if found is None:
        return call_real_asyncgen(*args, **kwargs), False

    construct, rule = found
    _mark_triggered(construct, canonical_key, rule)

    items = _get_generator_items_static(rule, canonical_key)
    if items is not None:
        return _iter_to_asyncgen(items), True

    result = rule.side_effect(*args, **kwargs)  # type: ignore[operator]
    if inspect.isawaitable(result):
        result = await result
    return await _ensure_asyncgen(result, canonical_key), True


async def _ensure_asyncgen(result: Any, canonical_key: str) -> AsyncGenerator[Any, None]:
    """Convert result to async generator or raise error.

    Returns:
        The async generator if valid.
    """
    if inspect.isasyncgen(result):
        return result
    if hasattr(result, "__aiter__"):
        return _aiter_to_asyncgen(result)
    if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
        return _iter_to_asyncgen(result)
    raise TenroSimulationExecutionError(
        canonical_key,
        expected_kind="async_generator",
        actual_kind=type(result).__name__,
    )


async def _aiter_to_asyncgen(ait: Any) -> AsyncGenerator[Any, None]:
    """Convert an async iterable to an async generator."""
    async for item in ait:
        yield item


async def _iter_to_asyncgen(it: Any) -> AsyncGenerator[Any, None]:
    """Convert a sync iterable to an async generator."""
    for item in it:
        yield item


__all__ = [
    "dispatch_async",
    "dispatch_asyncgen",
    "dispatch_gen",
    "dispatch_sync",
]
