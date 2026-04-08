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

_SUBMODULES = ("errors", "evals", "spans", "testing")


def _load_symbol(name: str) -> object | None:
    """Load a public symbol by name.

    Args:
        name: Symbol name to import.

    Returns:
        The imported symbol, or None if not found.
    """
    if name == "Construct":
        from tenro.construct import Construct

        return Construct
    if name == "EvalResult":
        from tenro.evals import EvalResult

        return EvalResult
    if name == "LLMResponse":
        from tenro.llm_response import LLMResponse

        return LLMResponse
    if name == "RawLLMResponse":
        from tenro.llm_response import RawLLMResponse

        return RawLLMResponse
    if name == "Provider":
        from tenro.providers import Provider

        return Provider
    if name == "ToolCall":
        from tenro.tool_calls import ToolCall

        return ToolCall
    if name == "init":
        from tenro._init import init

        return init
    if name == "tc":
        from tenro.tool_calls import tc

        return tc
    if name == "link_agent":
        from tenro.linking import link_agent

        return link_agent
    if name == "link_llm":
        from tenro.linking import link_llm

        return link_llm
    if name == "link_tool":
        from tenro.linking import link_tool

        return link_tool
    return None


def _load_submodule(name: str) -> object | None:
    """Load a submodule by name.

    Args:
        name: Submodule name to import.

    Returns:
        The imported module, or None if not found.
    """
    if name == "errors":
        import tenro.errors

        return tenro.errors
    if name == "evals":
        import tenro.evals

        return tenro.evals
    if name == "spans":
        import tenro.spans

        return tenro.spans
    if name == "testing":
        import tenro.testing

        return tenro.testing
    return None


def __getattr__(name: str) -> object:
    """Lazy-load public symbols and submodules on first access."""
    val = _load_symbol(name)
    if val is None:
        val = _load_submodule(name)
    if val is not None:
        globals()[name] = val
        return val

    if name == "__version__":
        from importlib.metadata import version

        v = version("tenro")
        globals()["__version__"] = v
        return v

    raise AttributeError(f"module 'tenro' has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available module attributes."""
    return sorted(list(__all__) + list(_SUBMODULES) + ["__version__"])


__all__ = [
    "Construct",
    "EvalResult",
    "LLMResponse",
    "Provider",
    "RawLLMResponse",
    "ToolCall",
    "init",
    "link_agent",
    "link_llm",
    "link_tool",
    "tc",
]
