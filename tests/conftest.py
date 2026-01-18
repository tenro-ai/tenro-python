# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for Tenro SDK tests."""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import anthropic
    import openai

    from tenro.construct import Construct

# Add tests/ to path so myapp and exampleapp can be imported
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def construct(request: pytest.FixtureRequest) -> Generator[Construct, None, None]:
    """Auto-activating construct fixture.

    The construct is automatically activated before the test and cleaned up after.
    No 'with' block needed - just use the construct directly!

    Args:
        request: Pytest fixture request for accessing test item.

    Yields:
        Pre-activated Construct instance.

    Example:
        >>> def test_my_agent(construct: Construct):
        ...     construct.simulate_llm(Provider.OPENAI, response="Hello")
        ...     result = my_agent.run()  # No 'with' needed!
        ...     assert "Hello" in result
    """
    from tenro.construct import Construct

    c = Construct()

    # Store reference on test item for trace visualization in teardown
    request.node._tenro_construct = c

    with c:
        yield c


@pytest.fixture
def openai_client() -> openai.OpenAI:
    """OpenAI client for testing."""
    from examples.myapp import get_openai_client

    return get_openai_client()


@pytest.fixture
def anthropic_client() -> anthropic.Anthropic:
    """Anthropic client for testing."""
    from examples.myapp import get_anthropic_client

    return get_anthropic_client()
