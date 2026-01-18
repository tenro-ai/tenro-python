# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pattern: Simulating errors to test failure handling.

Shows how to simulate failures and test your agent's error recovery.
"""

import pytest
from examples.myapp import ResilientAgent, call_api

from tenro.simulate import tool


def test_error_then_success(construct) -> None:
    """First call fails, retry succeeds - test recovery logic."""
    tool.simulate(
        call_api,
        results=[
            ConnectionError("Network timeout"),  # First call fails
            {"status": "ok", "data": [1, 2, 3]},  # Retry succeeds
        ],
    )

    result = ResilientAgent().run("fetch data")

    assert result["success"] is True
    tool.verify_many(call_api, count=2)  # Tried twice


def test_all_retries_fail(construct) -> None:
    """All retries fail - test error escalation."""
    tool.simulate(
        call_api,
        results=[
            ConnectionError("Attempt 1"),
            ConnectionError("Attempt 2"),
            ConnectionError("Attempt 3"),
        ],
    )

    with pytest.raises(ConnectionError):
        ResilientAgent().run("fetch data")

    tool.verify_many(call_api, count=3)  # All retries exhausted


def test_specific_error_types(construct) -> None:
    """Different error types trigger different handling."""
    tool.simulate(
        call_api,
        results=[
            ConnectionError("Connection refused"),
            {"status": "ok"},
        ],
    )

    result = ResilientAgent().run("fetch data")

    assert result["success"] is True
    tool.verify_many(call_api, count=2)
