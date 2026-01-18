# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Entry method constants for decorator target detection.

Centralizes the entry method lists for agents and tools across all supported
frameworks. These are used by the detection module to find wrappable methods.
"""

from __future__ import annotations

# Agent entry methods in precedence order (first match wins for primary entry point)
# Tuple provides deterministic ordering across Python versions
AGENT_ENTRY_PRECEDENCE: tuple[str, ...] = (
    # Generic
    "execute",
    "run",
    # LangChain
    "invoke",
    "ainvoke",
    # Dunder method
    "__call__",
    # CrewAI
    "kickoff",
    "kickoff_async",
    # AutoGen
    "initiate_chat",
    "initiate_chats",
    # LlamaIndex
    "chat",
    "achat",
    # Pydantic AI
    "run_sync",
    # Streaming methods
    "stream",
    "astream",
    "run_stream",
    "stream_chat",
)

# Frozenset for O(1) membership checks
AGENT_ENTRY_METHODS: frozenset[str] = frozenset(AGENT_ENTRY_PRECEDENCE)

# Tool entry methods in precedence order
TOOL_ENTRY_PRECEDENCE: tuple[str, ...] = (
    "invoke",
    "ainvoke",
    "run",
    "_run",
    "_arun",
    "__call__",
    # Streaming methods
    "stream",
    "astream",
)

# Frozenset for O(1) membership checks
TOOL_ENTRY_METHODS: frozenset[str] = frozenset(TOOL_ENTRY_PRECEDENCE)

__all__ = [
    "AGENT_ENTRY_METHODS",
    "AGENT_ENTRY_PRECEDENCE",
    "TOOL_ENTRY_METHODS",
    "TOOL_ENTRY_PRECEDENCE",
]
