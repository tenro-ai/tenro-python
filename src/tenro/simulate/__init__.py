# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Simulation module for LLM, tool, and agent testing.

Requires an active Construct context (via pytest fixture or context manager).

Examples:
    >>> from tenro.simulate import llm, tool, agent, register
    >>> from tenro import Provider
    >>>
    >>> def test_my_agent(construct):
    ...     llm.simulate(Provider.OPENAI, response="Hello!")
    ...     my_agent.run()
    ...     llm.verify()  # exactly 1 call
    >>>
    >>> # Register undecorated callables for simulation
    >>> from third_party import their_function
    >>> register(their_function)
"""

from __future__ import annotations

from tenro.providers import Provider
from tenro.tool_calls import ToolCall, tc

from . import agent, llm, tool
from ._register import register

__all__ = [
    "Provider",
    "ToolCall",
    "agent",
    "llm",
    "register",
    "tc",
    "tool",
]
