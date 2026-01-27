# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Sample agents and tools for testing.

Note: LLM functions requiring client libraries (openai) are in llm_agents.py
to avoid import errors in clean-room tests.
"""

from __future__ import annotations

from tenro import link_agent, link_tool


@link_tool
def search(query: str) -> list[str]:
    """Search tool."""
    raise NotImplementedError("Should be simulated")


@link_tool
def fetch(url: str) -> str:
    """Fetch tool."""
    raise NotImplementedError("Should be simulated")


@link_agent
class Researcher:
    """Research agent that finds information."""

    def run(self, task: str) -> str:
        """Execute research task."""
        raise NotImplementedError("Should be simulated")


@link_agent
class Writer:
    """Writer agent that generates content."""

    def run(self, prompt: str) -> str:
        """Execute writing task."""
        raise NotImplementedError("Should be simulated")
