# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pytest integration utilities.

Provides the @tenro decorator for enabling the construct fixture.

Examples:
    >>> from tenro.testing import tenro
    >>> from tenro.simulate import llm
    >>>
    >>> @tenro
    ... def test_my_agent():
    ...     llm.simulate(response="Hello!")
    ...     result = my_agent.run()
    ...     assert "Hello" in result
    >>>
    >>> @tenro
    ... class TestMyAgent:
    ...     def test_one(self): ...
    ...     def test_two(self): ...
"""

from __future__ import annotations

import pytest

tenro = pytest.mark.usefixtures("construct")
"""Pytest decorator that enables the construct fixture.

Equivalent to @pytest.mark.usefixtures("construct") but more concise.
Apply to test functions or classes to enable LLM/tool simulation.
"""

__all__ = ["tenro"]
