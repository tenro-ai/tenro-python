# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Output verification utilities.

Unified helpers for verifying output values:
- output: Smart match (dict=subset, scalar=exact)
- output_contains: Substring match (strings only)
- output_exact: Strict deep equality
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tenro._construct.verify.format import format_error_with_trace
from tenro.errors import TenroVerificationError
from tenro.util.list_helpers import normalize_and_validate_index


def is_subset_match(actual: Any, expected: Any) -> bool:
    """Check if actual contains expected.

    For dicts, performs deep subset matching. For other types, uses exact equality.

    Args:
        actual: The actual value to check.
        expected: The expected value or subset.

    Returns:
        `True` if `actual` contains `expected`.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(k in actual and is_subset_match(actual[k], v) for k, v in expected.items())
    return bool(actual == expected)


def _verify_output(
    items: list[Any],
    content_field: str,
    matcher: Callable[[Any], bool],
    expected: Any,
    item_type: str,
    *,
    call_index: int | None = 0,
) -> None:
    """Verify output using a custom matcher function.

    Args:
        items: List of items to check.
        content_field: Attribute name to read from each item.
        matcher: Predicate returning `True` on match.
        expected: Expected value for error messages.
        item_type: Label for items (e.g., "tool", "agent").
        call_index: Index to check (`None` checks all items).

    Raises:
        AssertionError: If no items exist or none match.
    """
    if not items:
        raise TenroVerificationError(f"No {item_type} calls found. Expected at least one.")

    if call_index is None:
        for item in items:
            if matcher(getattr(item, content_field, None)):
                return
        actuals = [getattr(item, content_field, None) for item in items]
        header = f"{item_type} output mismatch (checked all {len(items)} calls):"
        msg = format_error_with_trace(header, expected, actuals, span=None)
        raise TenroVerificationError(msg)

    item, _ = normalize_and_validate_index(items, call_index, item_type)
    actual = getattr(item, content_field, None)
    if matcher(actual):
        return
    header = f"{item_type}[{call_index}] output mismatch:"
    msg = format_error_with_trace(header, expected, actual, span=item)
    raise TenroVerificationError(msg)


def verify_output(
    items: list[Any],
    content_field: str,
    expected: Any,
    item_type: str,
    *,
    call_index: int | None = 0,
) -> None:
    """Verify output matches expected value.

    Uses subset matching for dicts and exact equality for scalars.

    Args:
        items: List of items to check.
        content_field: Attribute name to read from each item.
        expected: Expected value (dict=subset match, scalar=exact).
        item_type: Label for error messages (e.g., `"tool"`, `"agent"`).
        call_index: Which call to check (0=first, -1=last, `None`=any).

    Raises:
        AssertionError: If output doesn't match expected.
    """
    _verify_output(
        items,
        content_field,
        lambda actual: is_subset_match(actual, expected),
        expected,
        item_type,
        call_index=call_index,
    )


def verify_output_exact(
    items: list[Any],
    content_field: str,
    expected: Any,
    item_type: str,
    *,
    call_index: int | None = 0,
) -> None:
    """Verify output using strict deep equality.

    Args:
        items: List of items to check.
        content_field: Attribute name to read from each item.
        expected: Expected value (must match exactly).
        item_type: Label for error messages (e.g., `"tool"`, `"agent"`).
        call_index: Which call to check (0=first, -1=last, `None`=any).

    Raises:
        AssertionError: If output doesn't match expected exactly.
    """
    _verify_output(
        items,
        content_field,
        lambda actual: actual == expected,
        expected,
        item_type,
        call_index=call_index,
    )


def verify_output_contains(
    items: list[Any],
    content_field: str,
    expected_text: str,
    item_type: str,
    *,
    call_index: int | None = 0,
) -> None:
    """Verify output contains expected substring.

    Only works with string outputs.

    Args:
        items: List of items to check.
        content_field: Attribute name to read from each item.
        expected_text: Expected substring.
        item_type: Label for error messages (e.g., `"tool"`, `"agent"`).
        call_index: Which call to check (0=first, -1=last, `None`=any).

    Raises:
        AssertionError: If output doesn't contain expected substring.
    """
    _verify_output(
        items,
        content_field,
        lambda actual: isinstance(actual, str) and expected_text in actual,
        expected_text,
        item_type,
        call_index=call_index,
    )


__all__ = [
    "is_subset_match",
    "verify_output",
    "verify_output_contains",
    "verify_output_exact",
]
