# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Apply a TracePolicy to a captured value at export time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tenro.trace_policy.categories import Category
from tenro.trace_policy.modes import OMIT, CaptureMode, TransformContext
from tenro.trace_policy.policy import TracePolicy


@dataclass(frozen=True)
class CaptureResult:
    """Outcome of applying a policy to one captured value.

    Attributes:
        drop: If ``True``, the value must not be exported.
        value: Post-policy value. Only meaningful when ``drop`` is ``False``.
        error: Non-``None`` when the pipeline failed closed. Emit as
            ``tenro.trace.error`` event with this string.
    """

    drop: bool
    value: Any = None
    error: str | None = None


_DROP: CaptureResult = CaptureResult(drop=True)


def apply(
    policy: TracePolicy,
    value: Any,
    *,
    category: Category,
    span_type: str = "other",
    span_name: str = "",
    attribute: str = "",
) -> CaptureResult:
    """Apply ``policy`` to ``value`` for the given category.

    Args:
        policy: Resolved trace policy.
        value: Raw captured value.
        category: Data category.
        span_type: Span kind (``llm``, ``tool``, ``agent``, ``retrieval``).
        span_name: Span name (tool name, model name).
        attribute: Dotted attribute path within the span.

    Returns:
        ``CaptureResult`` with either the emitted value or a drop flag.
        A non-``Category`` argument fails closed (drop + error stamp)
        so bad instrumentation never aborts span export.
    """
    if not isinstance(category, Category):
        return CaptureResult(
            drop=True,
            error=f"invalid_category: {type(category).__name__}",
        )
    if not policy.captures(category):
        return _DROP
    if policy.capture is CaptureMode.FULL:
        return CaptureResult(drop=False, value=value)
    # CUSTOM — TracePolicy.__post_init__ guarantees transform is not None
    # and policy.captures() already rejected OFF above.
    assert policy.transform is not None
    ctx = TransformContext(
        category=category,
        span_type=span_type,
        span_name=span_name,
        attribute=attribute,
    )
    try:
        result = policy.transform(value, ctx)
    except Exception as exc:
        return CaptureResult(drop=True, error=f"{category.value}: {type(exc).__name__}")
    if result is OMIT:
        return _DROP
    return CaptureResult(drop=False, value=result)


__all__ = ["CaptureResult", "apply"]
