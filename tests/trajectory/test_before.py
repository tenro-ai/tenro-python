# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Tests for verify_tool_before."""

from __future__ import annotations

import pytest

from tenro._construct.verify.trajectory import verify_tool_before
from tenro.errors import TenroValidationError, TenroVerificationError

from .conftest import calls


class TestVerifyToolBefore:
    def test_correct_order(self) -> None:
        verify_tool_before(calls("search", "summarize"), "search", "summarize", "tool")

    def test_correct_order_with_extras(self) -> None:
        verify_tool_before(calls("search", "fetch", "summarize"), "search", "summarize", "tool")

    def test_existential_with_repeated_tools(self) -> None:
        verify_tool_before(calls("summarize", "search", "summarize"), "search", "summarize", "tool")

    def test_wrong_order_fails(self) -> None:
        with pytest.raises(TenroVerificationError, match="no such ordering"):
            verify_tool_before(calls("summarize", "search"), "search", "summarize", "tool")

    def test_earlier_missing_fails(self) -> None:
        with pytest.raises(TenroVerificationError, match="'search' was never called"):
            verify_tool_before(calls("summarize", "fetch"), "search", "summarize", "tool")

    def test_later_missing_fails(self) -> None:
        with pytest.raises(TenroVerificationError, match="'summarize' was never called"):
            verify_tool_before(calls("search", "fetch"), "search", "summarize", "tool")

    def test_empty_calls_fails(self) -> None:
        with pytest.raises(TenroVerificationError, match="never called"):
            verify_tool_before([], "search", "summarize", "tool")

    def test_same_tool_rejects(self) -> None:
        with pytest.raises(TenroValidationError, match="different tools"):
            verify_tool_before(calls("search", "search"), "search", "search", "tool")

    def test_diagnostic_shows_trajectory(self) -> None:
        with pytest.raises(TenroVerificationError, match=r"Actual trajectory:.*summarize.*search"):
            verify_tool_before(calls("summarize", "search"), "search", "summarize", "tool")
