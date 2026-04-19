# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Tests for verify_tool_subsequence."""

from __future__ import annotations

import pytest

from tenro._construct.verify.trajectory import verify_tool_subsequence
from tenro.errors import TenroValidationError, TenroVerificationError

from .conftest import calls


class TestVerifyToolSubsequence:
    def test_exact_match(self) -> None:
        verify_tool_subsequence(calls("search", "summarize"), ["search", "summarize"], "tool")

    def test_extras_between(self) -> None:
        verify_tool_subsequence(
            calls("search", "fetch", "summarize"), ["search", "summarize"], "tool"
        )

    def test_extras_before_and_after(self) -> None:
        verify_tool_subsequence(
            calls("init", "search", "fetch", "summarize", "cleanup"),
            ["search", "summarize"],
            "tool",
        )

    def test_single_element(self) -> None:
        verify_tool_subsequence(calls("search", "fetch"), ["search"], "tool")

    def test_missing_tool_fails(self) -> None:
        with pytest.raises(TenroVerificationError, match="Not found after prefix"):
            verify_tool_subsequence(calls("search", "fetch"), ["search", "summarize"], "tool")

    def test_wrong_order_fails(self) -> None:
        with pytest.raises(TenroVerificationError, match="Not found after prefix"):
            verify_tool_subsequence(calls("summarize", "search"), ["search", "summarize"], "tool")

    def test_empty_calls_fails(self) -> None:
        with pytest.raises(TenroVerificationError, match="Not found"):
            verify_tool_subsequence([], ["search"], "tool")

    def test_empty_expected_raises_validation(self) -> None:
        with pytest.raises(TenroValidationError, match="empty"):
            verify_tool_subsequence(calls("search"), [], "tool")

    def test_diagnostic_shows_matched_prefix(self) -> None:
        with pytest.raises(TenroVerificationError, match=r"Matched prefix: \['a', 'b'\]"):
            verify_tool_subsequence(calls("a", "b", "x"), ["a", "b", "c"], "tool")

    def test_diagnostic_shows_trajectory(self) -> None:
        with pytest.raises(TenroVerificationError, match=r"Actual trajectory:.*search.*fetch"):
            verify_tool_subsequence(calls("search", "fetch"), ["search", "summarize"], "tool")
