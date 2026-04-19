# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Trajectory verification: subsequence and ordering assertions."""

from __future__ import annotations

from typing import Any, Literal

from tenro.errors import TenroValidationError, TenroVerificationError

TrajectoryEventType = Literal["tool", "agent"]


def _get_names(calls: Any, event_type: TrajectoryEventType) -> list[str]:
    """Extract display names from calls. Raises if any item lacks display_name."""
    names: list[str] = []
    for c in calls:
        if not hasattr(c, "display_name"):
            raise TenroValidationError(f"Call object {c!r} has no display_name attribute")
        names.append(c.display_name)
    return names


def _subsequence_match_length(actual: list[str], expected: list[str]) -> int:
    """Return how many elements of expected matched as a subsequence of actual."""
    idx = 0
    for name in actual:
        if idx < len(expected) and name == expected[idx]:
            idx += 1
    return idx


def verify_tool_subsequence(
    calls: Any, expected: list[str], event_type: TrajectoryEventType
) -> None:
    """Verify expected names appear as a subsequence (in order, extras allowed).

    Raises:
        TenroValidationError: If expected is empty.
        TenroVerificationError: If subsequence not found.
    """
    if not expected:
        raise TenroValidationError("expected must not be empty")

    actual = _get_names(calls, event_type)
    matched = _subsequence_match_length(actual, expected)

    if matched >= len(expected):
        return

    raise TenroVerificationError(
        f"Expected subsequence {expected} not found.\n"
        f"Matched prefix: {expected[:matched]}\n"
        f"Not found after prefix: {expected[matched:]}\n"
        f"Actual trajectory: {actual}"
    )


def _assert_tool_present(name: str, actual: list[str]) -> None:
    if name not in actual:
        raise TenroVerificationError(
            f"Tool '{name}' was never called.\nActual trajectory: {actual}"
        )


def verify_tool_before(
    calls: Any, earlier: str, later: str, event_type: TrajectoryEventType
) -> None:
    """Verify earlier appears before later (existential — any occurrence pair).

    Raises:
        TenroVerificationError: If ordering not found or tool missing.
    """
    if earlier == later:
        raise TenroValidationError(
            f"earlier and later must be different tools, got '{earlier}' for both"
        )

    actual = _get_names(calls, event_type)
    _assert_tool_present(earlier, actual)
    _assert_tool_present(later, actual)

    if _subsequence_match_length(actual, [earlier, later]) < 2:
        raise TenroVerificationError(
            f"Expected '{earlier}' before '{later}', but no such ordering found.\n"
            f"Actual trajectory: {actual}"
        )


def verify_tools_called(
    calls: Any,
    names: set[str] | list[str],
    event_type: TrajectoryEventType,
) -> None:
    """Verify all named tools were called at least once, any order, extras allowed.

    Raises:
        TenroValidationError: If names is empty.
        TenroVerificationError: If any named tool is missing.
    """
    if not names:
        raise TenroValidationError("names must not be empty")

    expected = set(names)
    actual = _get_names(calls, event_type)
    actual_set = set(actual)
    missing = expected - actual_set

    if missing:
        raise TenroVerificationError(
            f"Expected tools {sorted(expected)} to all be called.\n"
            f"Missing: {sorted(missing)}\n"
            f"Called:  {sorted(actual_set)}\n"
            f"Actual trajectory: {actual}"
        )
