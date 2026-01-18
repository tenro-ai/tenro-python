# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Target type detection for linking decorators.

Classifies decorator targets as FUNCTION, CLASS, or FRAMEWORK_OBJECT
to determine the appropriate wrapping strategy.

"""

from __future__ import annotations

import inspect
from enum import Enum, auto
from typing import Any


class TargetType(Enum):
    """Classification of decorator target."""

    FUNCTION = auto()
    CLASS = auto()
    FRAMEWORK_OBJECT = auto()


def detect_target_type(target: Any) -> TargetType:
    """Detect the type of decorator target.

    Detection priority:
    1. isclass -> CLASS
    2. Has callable invoke/run method -> FRAMEWORK_OBJECT
    3. callable -> FUNCTION

    Args:
        target: The decorated target (function, class, or object).

    Returns:
        TargetType classification.

    Raises:
        TypeError: If target is not callable.
    """
    if inspect.isclass(target):
        return TargetType.CLASS

    if _is_framework_object(target):
        return TargetType.FRAMEWORK_OBJECT

    if callable(target):
        return TargetType.FUNCTION

    raise TypeError(f"Cannot decorate {type(target).__name__}: not callable")


def _is_framework_object(target: Any) -> bool:
    """Check if target is a framework object (e.g., LangChain StructuredTool).

    Framework objects are instances (not classes) that have callable
    invoke/run methods or __call__, indicating pre-constructed tools/agents.
    """
    # Must not be a class itself
    if inspect.isclass(target):
        return False

    # Functions are not framework objects (handled separately)
    if inspect.isfunction(target) or inspect.ismethod(target):
        return False

    # Function wrappers (e.g., @cache, @lru_cache) have __wrapped__ - treat as functions
    if hasattr(target, "__wrapped__"):
        return False

    from tenro.linking.constants import AGENT_ENTRY_METHODS, TOOL_ENTRY_METHODS

    entry_methods = (AGENT_ENTRY_METHODS | TOOL_ENTRY_METHODS) - {"__call__"}
    for method_name in entry_methods:
        method = getattr(target, method_name, None)
        if method is not None and callable(method):
            return True

    # Check __call__ only if explicitly defined AND no standard entry methods exist
    obj_type = type(target)
    return "__call__" in obj_type.__dict__ and obj_type.__module__ not in ("functools", "builtins")


__all__ = [
    "TargetType",
    "_is_framework_object",
    "detect_target_type",
]
