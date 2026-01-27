# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Agent simulation tests."""

from __future__ import annotations

from myapp.agents import Researcher, Writer

from tenro.simulate import agent
from tenro.testing import tenro


class TestAgentSimulation:
    """Agent simulation and verification."""

    @tenro
    def test_single_result(self) -> None:
        """Simulate an agent returning a single result."""
        agent.simulate(Researcher, result="Research complete")

        result = Researcher().run("Find information")

        assert result == "Research complete"
        run = agent.verify(Researcher)
        assert run.simulated is True

    @tenro
    def test_sequential_results(self) -> None:
        """Simulate an agent returning different results on each call."""
        agent.simulate(Writer, results=["Draft 1", "Draft 2"])

        writer = Writer()
        assert writer.run("Write intro") == "Draft 1"
        assert writer.run("Write conclusion") == "Draft 2"

        runs = agent.verify_many(Writer, count=2)
        assert runs[0].output_data == "Draft 1"
        assert runs[1].output_data == "Draft 2"
        assert all(r.simulated for r in runs)
