# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Tool simulation tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from myapp.agents import fetch, search

from tenro.simulate import tool

if TYPE_CHECKING:
    from tenro import Construct


class TestToolSimulation:
    """Tool simulation and verification."""

    def test_single_result(self, construct: Construct) -> None:
        """Simulate a tool returning a single result."""
        tool.simulate(search, result=["result1", "result2"])

        result = search("test query")

        assert result == ["result1", "result2"]
        calls = tool.verify_many(search, count=1)
        assert calls[0].simulated is True

    def test_sequential_results(self, construct: Construct) -> None:
        """Simulate a tool returning different results on each call."""
        tool.simulate(fetch, results=["page1", "page2"])

        assert fetch("http://a.com") == "page1"
        assert fetch("http://b.com") == "page2"

        calls = tool.verify_many(fetch, count=2)
        assert calls[0].result == "page1"
        assert calls[1].result == "page2"
        assert all(c.simulated for c in calls)
