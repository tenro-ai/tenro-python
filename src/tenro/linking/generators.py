# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Generator wrappers for streaming span support.

Keeps spans open during generator iteration so that yielded values
are tracked under the correct parent span.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import Token
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from tenro._core.lifecycle_manager import LifecycleManager
    from tenro._core.spans import BaseSpan
    from tenro.linking.context import GuardKey

T = TypeVar("T")

# Type alias for guard token
GuardToken = Token[frozenset["GuardKey"]] | None
GuardExitFn = Callable[[GuardToken], None]


def wrap_generator(
    gen: Generator[T, None, None],
    span: BaseSpan,
    parent_span_id: str | None,
    lifecycle: LifecycleManager,
    guard_token: GuardToken,
    guard_exit_fn: GuardExitFn,
) -> Generator[T, None, None]:
    """Wrap a sync generator to keep its span open during iteration.

    The span closes when the generator completes, raises an exception, or
    is closed early (via break or explicit close).

    Args:
        gen: Original generator to wrap.
        span: Span tracking this generator.
        parent_span_id: Parent span ID for event linking.
        lifecycle: Manager for span start/end events.
        guard_token: Token for re-entrancy guard cleanup.
        guard_exit_fn: Callback to release the guard.

    Yields:
        Items from the wrapped generator.
    """
    error: Exception | None = None
    try:
        yield from gen
    except GeneratorExit:
        raise  # Early exit (break/close) completes normally
    except Exception as e:
        error = e
        raise
    finally:
        try:
            if error is not None:
                lifecycle.error_span_manual(span, parent_span_id, error)
            else:
                lifecycle.end_span_manual(span, parent_span_id)
        finally:
            guard_exit_fn(guard_token)


async def wrap_async_generator(
    gen: AsyncGenerator[T, None],
    span: BaseSpan,
    parent_span_id: str | None,
    lifecycle: LifecycleManager,
    guard_token: GuardToken,
    guard_exit_fn: GuardExitFn,
) -> AsyncGenerator[T, None]:
    """Wrap an async generator to keep its span open during iteration.

    The span closes when the generator completes, raises an exception, or
    is closed early (via break or explicit aclose).

    Args:
        gen: Original async generator to wrap.
        span: Span tracking this generator.
        parent_span_id: Parent span ID for event linking.
        lifecycle: Manager for span start/end events.
        guard_token: Token for re-entrancy guard cleanup.
        guard_exit_fn: Callback to release the guard.

    Yields:
        Items from the wrapped async generator.
    """
    error: Exception | None = None
    try:
        async for item in gen:
            yield item
    except GeneratorExit:
        raise
    except Exception as e:
        error = e
        raise
    finally:
        try:
            if error is not None:
                lifecycle.error_span_manual(span, parent_span_id, error)
            else:
                lifecycle.end_span_manual(span, parent_span_id)
        finally:
            guard_exit_fn(guard_token)
