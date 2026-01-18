# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Content verification utilities for assertions.

Provides helpers for verifying content in lists of items.
"""

from __future__ import annotations

from typing import TypeVar

from tenro.errors import TenroVerificationError
from tenro.util.list_helpers import normalize_and_validate_index

T = TypeVar("T")


def verify_content_contains(
    items: list[T],
    content_field: str,
    expected_text: str,
    item_type: str,
    *,
    call_index: int | None = 0,
) -> None:
    """Verify that item content contains expected text.

    Supports three modes:
    - call_index=0 (default): Check first item
    - call_index=N: Check specific item (supports negative indexing)
    - call_index=None: Check ALL items (permissive mode)

    Args:
        items: List of items to check.
        content_field: Name of field to check (e.g., "response", "value",
            "output_data").
        expected_text: Text to search for.
        item_type: Human-readable name for error messages (e.g., "LLM response",
            "tool execution").
        call_index: Index of specific item to check (0-based, default=0).
            Supports negative indexing: -1 = last, -2 = second to last.
            Set to None to check all items (permissive mode).

    Raises:
        AssertionError: If content doesn't contain expected text or items list is empty.

    Examples:
        >>> calls = [Simulation(response="Hello"), Simulation(response="World")]
        >>> verify_content_contains(calls, "response", "Hello", "LLM response")
        >>> verify_content_contains(
        ...     calls, "response", "World", "LLM response", call_index=-1
        ... )
        >>> verify_content_contains(
        ...     calls, "response", "o", "LLM response", call_index=None
        ... )
    """
    if not items:
        msg = f"Expected {item_type} containing '{expected_text}', "
        msg += f"but no {item_type}s found."
        raise TenroVerificationError(msg)

    if call_index is None:
        for item in items:
            content = getattr(item, content_field, None)
            if content and expected_text in content:
                return

        msg = f"Expected {item_type} containing '{expected_text}', "
        msg += f"but no {item_type}s matched."
        raise TenroVerificationError(msg)

    item, _normalized_index = normalize_and_validate_index(items, call_index, item_type)

    content = getattr(item, content_field, None)
    if content and expected_text in content:
        return

    msg = f"Expected {item_type} at index {call_index} to contain "
    msg += f"'{expected_text}', but it doesn't."
    raise TenroVerificationError(msg)


__all__ = ["verify_content_contains"]
