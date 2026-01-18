# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Agent simulation tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from myapp.agents import researcher, writer

from tenro.simulate import agent

if TYPE_CHECKING:
    from tenro import Construct


class TestAgentSimulation:
    """Agent simulation and verification."""

    def test_single_result(self, construct: Construct) -> None:
        """Simulate an agent returning a single result."""
        agent.simulate(researcher, result="Research complete")

        result = researcher("Find information")

        assert result == "Research complete"
        call = agent.verify(researcher)
        assert call.simulated is True

    def test_sequential_results(self, construct: Construct) -> None:
        """Simulate an agent returning different results on each call."""
        agent.simulate(writer, results=["Draft 1", "Draft 2"])

        assert writer("Write intro") == "Draft 1"
        assert writer("Write conclusion") == "Draft 2"

        calls = agent.verify_many(writer, count=2)
        assert calls[0].output_data == "Draft 1"
        assert calls[1].output_data == "Draft 2"
        assert all(c.simulated for c in calls)
