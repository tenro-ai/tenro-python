# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Tests for verify_tools_called."""

from __future__ import annotations

import pytest

from tenro._construct.verify.trajectory import verify_tools_called
from tenro.errors import TenroValidationError, TenroVerificationError

from .conftest import calls


class TestVerifyToolsCalled:
    def test_all_present(self) -> None:
        verify_tools_called(calls("search", "fetch", "summarize"), {"search", "summarize"}, "tool")

    def test_extras_allowed(self) -> None:
        verify_tools_called(calls("init", "search", "summarize", "cleanup"), {"search"}, "tool")

    def test_order_ignored(self) -> None:
        verify_tools_called(calls("summarize", "search"), {"search", "summarize"}, "tool")

    def test_missing_tool_fails(self) -> None:
        with pytest.raises(TenroVerificationError, match=r"Missing.*summarize"):
            verify_tools_called(calls("search", "fetch"), {"search", "summarize"}, "tool")

    def test_empty_calls_fails(self) -> None:
        with pytest.raises(TenroVerificationError, match="Missing"):
            verify_tools_called([], {"search"}, "tool")

    def test_empty_names_raises_validation(self) -> None:
        with pytest.raises(TenroValidationError, match="empty"):
            verify_tools_called(calls("search"), set(), "tool")

    def test_diagnostic_shows_called_and_missing(self) -> None:
        with pytest.raises(TenroVerificationError, match=r"Called:.*search"):
            verify_tools_called(calls("search"), {"search", "missing_tool"}, "tool")
