# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Metadata protocol and helpers for span identity resolution.

Provides utilities for classifying targets, detecting decorated
callables, and finding entry methods in precedence order.
"""

from __future__ import annotations

import inspect
from typing import Any, Literal

from tenro.linking.constants import AGENT_ENTRY_PRECEDENCE, TOOL_ENTRY_PRECEDENCE

# Metadata attribute names stored on decorated targets
ATTR_LINKED_TYPE = "_tenro_linked_type"
ATTR_TARGET_PATHS = "_tenro_target_paths"
ATTR_DISPLAY_NAME = "_tenro_display_name"
ATTR_SOURCE_NAME = "_tenro_source_name"
ATTR_TARGET_KIND = "_tenro_target_kind"
ATTR_LINKED_METHODS = "_tenro_linked_methods"
ATTR_WRAPPED = "__tenro_wrapped__"

TargetKind = Literal["class", "function", "object", "string", "bound_method"]


def classify_target(target: Any) -> TargetKind:
    """Classify target type for dispatch.

    Distinguishes between classes, functions, bound methods, callable
    objects, and strings. Uses inspect module checks rather than
    callable() to correctly classify callable instances.

    Args:
        target: The target to classify (class, function, object, or string).

    Returns:
        One of: "string", "class", "function", "bound_method", "object".
    """
    if isinstance(target, str):
        return "string"
    if isinstance(target, type):
        return "class"
    if inspect.isfunction(target):
        return "function"
    if inspect.ismethod(target):
        return "bound_method"
    # Includes callable instances with __call__ - NOT using callable()
    return "object"


def is_directly_linked(cls: type) -> bool:
    """Check if class itself (not parent) is decorated.

    Uses cls.__dict__ instead of hasattr() to avoid detecting
    inherited metadata from parent classes.

    Args:
        cls: The class to check.

    Returns:
        True if cls has _tenro_linked_type in its own __dict__.
    """
    return ATTR_LINKED_TYPE in cls.__dict__


def _is_explicitly_defined(cls: type, method_name: str) -> bool:
    """Check if a method is explicitly defined (not inherited from object/type).

    This is used to filter out __call__ inherited from type/object, which
    all classes have but shouldn't be considered an entry method unless
    explicitly defined.
    """
    for klass in cls.__mro__:
        if method_name in klass.__dict__:
            return klass is cls or klass not in (object, type)
    return False


def find_entry_methods(
    cls: type,
    precedence: tuple[str, ...] = AGENT_ENTRY_PRECEDENCE,
) -> list[str]:
    """Find entry methods on a class in precedence order.

    Returns methods in deterministic order based on the precedence
    tuple, ensuring consistent behavior across Python versions.

    Filters out __call__ unless explicitly defined on the class or
    a non-builtin parent class (to avoid matching the inherited
    __call__ from type/object).

    Args:
        cls: The class to inspect.
        precedence: Tuple of method names in priority order.
            Defaults to AGENT_ENTRY_PRECEDENCE.

    Returns:
        List of method names found on cls, in precedence order.
    """
    found: list[str] = []
    for method_name in precedence:
        if not hasattr(cls, method_name):
            continue
        # Skip inherited object.__call__ if not explicitly defined
        if method_name == "__call__" and not _is_explicitly_defined(cls, method_name):
            continue
        # Skip property descriptors - calling them executes the getter
        # staticmethod/classmethod ARE valid entry methods
        raw_attr = _get_raw_attribute(cls, method_name)
        if isinstance(raw_attr, property):
            continue
        # Verify the attribute is callable
        attr = getattr(cls, method_name)
        if not callable(attr):
            continue
        found.append(method_name)
    return found


def _get_raw_attribute(cls: type, name: str) -> Any:
    """Get the raw attribute from the class __dict__ (before descriptor protocol)."""
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass.__dict__[name]
    return None


def find_tool_entry_methods(cls: type) -> list[str]:
    """Find tool entry methods on a class in precedence order.

    Args:
        cls: The class to inspect.

    Returns:
        List of tool method names found on cls, in precedence order.
    """
    return find_entry_methods(cls, TOOL_ENTRY_PRECEDENCE)


def get_target_paths(target: Any) -> tuple[str, ...] | None:
    """Get stored target paths from a decorated target.

    Args:
        target: A potentially decorated callable.

    Returns:
        The target paths tuple if present, None otherwise.
    """
    paths = getattr(target, ATTR_TARGET_PATHS, None)
    if paths is None:
        return None
    if isinstance(paths, str):
        raise ValueError(
            f"{ATTR_TARGET_PATHS} must be tuple[str, ...], not str. Use (path,) for singletons."
        )
    return tuple(paths)


def get_display_name(target: Any) -> str | None:
    """Get stored display name from a decorated target.

    Args:
        target: A potentially decorated callable.

    Returns:
        The display name if present, None otherwise.
    """
    return getattr(target, ATTR_DISPLAY_NAME, None)


__all__ = [
    "ATTR_DISPLAY_NAME",
    "ATTR_LINKED_METHODS",
    # Attribute names
    "ATTR_LINKED_TYPE",
    "ATTR_SOURCE_NAME",
    "ATTR_TARGET_KIND",
    "ATTR_TARGET_PATHS",
    "ATTR_WRAPPED",
    # Types
    "TargetKind",
    # Functions
    "classify_target",
    "find_entry_methods",
    "find_tool_entry_methods",
    "get_display_name",
    "get_target_paths",
    "is_directly_linked",
]
