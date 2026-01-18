# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Construct class for LLM testing.

Provides the `Construct` class for simulating and verifying LLM, tool,
and agent calls in tests.
"""

from __future__ import annotations

from tenro._construct.construct import Construct as _ConstructImpl


class Construct(_ConstructImpl):
    """Test harness for simulating and verifying LLM, tool, and agent calls.

    Use as a pytest fixture or context manager:

        def test_example(construct):
            construct.simulate_llm(Provider.OPENAI, response="Hello")
            # ... run agent code ...
            construct.verify_llm(Provider.OPENAI)

    For manual context management (non-pytest):

        async with Construct() as construct:
            construct.simulate_llm(Provider.ANTHROPIC, response="Hi")
            await my_agent.run()
            construct.verify_llm()
    """

    pass


__all__ = ["Construct"]
