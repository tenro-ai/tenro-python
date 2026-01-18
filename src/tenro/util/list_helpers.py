# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""List manipulation and validation utilities.

Provides helpers for working with lists in verification logic.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def normalize_and_validate_index(
    items: list[T],
    index: int,
    item_type: str,
) -> tuple[T, int]:
    """Normalize negative index and validate bounds.

    Supports Python-style negative indexing (e.g., `-1` for last item).

    Args:
        items: List to index into.
        index: Index (supports negative indexing: -1 = last, -2 = second to last).
        item_type: Human-readable name for error messages (e.g., "LLM call").

    Returns:
        Tuple of (item at index, normalized positive index).

    Raises:
        AssertionError: If index is out of bounds.

    Examples:
        >>> calls = ["first", "second", "third"]
        >>> item, idx = normalize_and_validate_index(calls, 0, "call")
        >>> assert item == "first" and idx == 0
        >>>
        >>> item, idx = normalize_and_validate_index(calls, -1, "call")
        >>> assert item == "third" and idx == 2
    """
    original_index = index
    if index < 0:
        index = len(items) + index

    if index < 0 or index >= len(items):
        msg = f"Invalid call_index={original_index}. "
        msg += f"Only {len(items)} {item_type}(s) recorded."
        raise AssertionError(msg)

    return items[index], index


__all__ = ["normalize_and_validate_index"]
