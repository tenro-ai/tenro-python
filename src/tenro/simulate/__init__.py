# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Simulation module for LLM, tool, and agent testing.

This module doubles as a pytest decorator. Use ``@tenro.simulate`` on test
functions or classes to enable the Construct fixture automatically::

    import tenro

    @tenro.simulate
    def test_my_agent():
        llm.simulate(response="Hello!")
        ...

    @tenro.simulate
    class TestMyAgent:
        def test_one(self): ...

Parenthesised form works with or without options::

    @tenro.simulate()
    def test_my_agent(): ...

    @tenro.simulate(allow_real_llm_calls=True)
    def test_real_llm(): ...

Submodule imports are unaffected::

    from tenro.simulate import llm, tool, agent, register
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from typing import Any, TypeVar, overload

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

_T = TypeVar("_T")


def _apply_markers(obj: _T, **options: Any) -> _T:
    """Apply construct fixture and optional tenro marker to a test target.

    Imports pytest lazily so ``from tenro.simulate import llm`` works
    in production environments where pytest is not installed.

    Args:
        obj: Test function or class.
        **options: Options forwarded as ``pytest.mark.tenro`` keyword args.

    Returns:
        Decorated test target.
    """
    import pytest

    result = pytest.mark.usefixtures("construct")(obj)
    if options:
        result = pytest.mark.tenro(**options)(result)
    return result  # type: ignore[return-value]


class _SimulateModule(types.ModuleType):
    """Module subclass that makes ``tenro.simulate`` callable as a decorator.

    Supports three forms::

        @tenro.simulate           # bare
        @tenro.simulate()         # empty parens
        @tenro.simulate(opt=val)  # with options
    """

    @overload
    def __call__(self, __obj: _T) -> _T: ...

    @overload
    def __call__(self, **options: Any) -> Callable[[_T], _T]: ...

    def __call__(self, __obj: _T | None = None, **options: Any) -> _T | Callable[[_T], _T]:
        if __obj is not None:
            # Bare form: @tenro.simulate
            return _apply_markers(__obj, **options)

        # Parenthesised form: @tenro.simulate() or @tenro.simulate(opt=val)
        def decorator(target: _T) -> _T:
            return _apply_markers(target, **options)

        return decorator


sys.modules[__name__].__class__ = _SimulateModule
