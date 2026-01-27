# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Tenro Python SDK for local AI agent testing and evaluation.

Examples:
    >>> from tenro import Construct, Provider, ToolCall
    >>> from tenro import link_llm, link_tool, link_agent
    >>> from tenro.simulate import llm, tool, agent
    >>> from tenro.errors import TenroError
    >>> from tenro.spans import LLMCallSpan, ToolCallSpan, AgentRunSpan
"""

from __future__ import annotations

from importlib.metadata import version

from tenro.construct import Construct
from tenro.evals import EvalResult
from tenro.linking import link_agent, link_llm, link_tool
from tenro.llm_response import LLMResponse, RawLLMResponse
from tenro.providers import Provider
from tenro.tool_calls import ToolCall, tc

__version__ = version("tenro")

_SUBMODULES = {"errors", "evals", "spans", "testing"}


def __getattr__(name: str) -> object:
    """Import submodules on attribute access."""
    # Explicit branches to avoid dynamic import_module(variable)
    if name == "errors":
        import tenro.errors as errors

        return errors
    if name == "evals":
        import tenro.evals as evals

        return evals
    if name == "spans":
        import tenro.spans as spans

        return spans
    if name == "testing":
        import tenro.testing as testing

        return testing
    raise AttributeError(f"module 'tenro' has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available module attributes."""
    return sorted(list(__all__) + list(_SUBMODULES))


__all__ = [
    "Construct",
    "EvalResult",
    "LLMResponse",
    "Provider",
    "RawLLMResponse",
    "ToolCall",
    "link_agent",
    "link_llm",
    "link_tool",
    "tc",
]
