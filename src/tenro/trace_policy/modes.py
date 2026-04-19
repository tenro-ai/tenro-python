# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Trace policy capture modes and transform types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from tenro.trace_policy.categories import Category


class CaptureMode(StrEnum):
    """Top-level capture mode for a trace policy.

    Attributes:
        OFF: Drop all content. Structural data (timings, token counts, span
            graph) still emits. Default outside test context.
        FULL: Emit content as-is. Default inside pytest / ``@simulate``.
        CUSTOM: Delegate to a ``Transform`` callback per category.
    """

    OFF = "off"
    FULL = "full"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class TransformContext:
    """Context supplied to a ``Transform`` callback.

    Attributes:
        category: The ``Category`` of the value being transformed.
        span_type: Span kind identifier (``llm``, ``tool``, ``agent``,
            ``retrieval``, or ``other``).
        span_name: Human-readable span name (tool name, model name, agent name).
        attribute: Dotted attribute path within the span (e.g. ``messages.0.content``).
    """

    category: Category
    span_type: str = "other"
    span_name: str = ""
    attribute: str = ""


class _OmitSentinel:
    """Sentinel return value telling the applier to drop the value."""

    _instance: _OmitSentinel | None = None

    def __new__(cls) -> _OmitSentinel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "OMIT"

    def __bool__(self) -> bool:
        return False


OMIT: Final = _OmitSentinel()
"""Return ``OMIT`` from a ``Transform`` to drop the value."""


Transform = Callable[[Any, TransformContext], Any]
"""Callable type for ``capture=CUSTOM`` callbacks.

Receives the raw value and a ``TransformContext``. Return the transformed
value or ``OMIT`` to drop it. Exceptions drop the value and fire a
``tenro.trace.error`` event.
"""


__all__ = [
    "OMIT",
    "CaptureMode",
    "Transform",
    "TransformContext",
]
