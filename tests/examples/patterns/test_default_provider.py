# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pattern: Setting a default provider.

Shows how to set a default provider so you don't have to specify it
on every simulate/verify call. Two approaches are shown:

1. Fixture-based: Create a custom fixture that sets the default for all tests
2. Per-test: Call construct.set_default_provider() in each test
"""

from __future__ import annotations

import pytest
from examples.myapp import call_llm, multi_step_workflow

from tenro import Construct, Provider
from tenro.simulate import llm
from tenro.testing import tenro

# =============================================================================
# APPROACH 1: Fixture-based default provider
# Recommended when using a single LLM provider across all tests
# =============================================================================


@pytest.fixture
def tenro_construct(construct: Construct) -> Construct:
    """Construct fixture with OpenAI as the default provider.

    Use this fixture instead of 'construct' when all tests in a module
    use the same provider. This avoids repetition and ensures consistency.

    Note: Only recommended if your application uses a single LLM provider.
    If you use multiple providers (e.g., OpenAI + Anthropic), specify the
    provider explicitly on each simulate/verify call instead.

    For module-wide usage, add this to your conftest.py:

        @pytest.fixture
        def construct(construct: Construct) -> Construct:
            construct.set_default_provider(Provider.OPENAI)
            return construct
    """
    construct.set_default_provider(Provider.OPENAI)
    return construct


def test_with_fixture_default_provider(tenro_construct: Construct) -> None:
    """Using fixture-based default provider - cleanest approach."""
    # No need to set provider - fixture already did it
    llm.simulate(
        responses=[
            "Research results",
            "Summary",
            "Recommendations",
        ]
    )

    multi_step_workflow("AI trends")

    # Verify also works without provider
    llm.verify_many(count=3)


def test_fixture_with_override(tenro_construct: Construct) -> None:
    """You can still override the default when needed."""
    # Uses default (OpenAI)
    llm.simulate(response="OpenAI response")

    call_llm("test")

    llm.verify()

    # To use a different provider, just specify it explicitly:
    # llm.simulate(Provider.ANTHROPIC, response="Anthropic response")


# =============================================================================
# APPROACH 2: Per-test default provider
# =============================================================================


@tenro
def test_without_default_provider() -> None:
    """Without default provider, you must specify provider on every call."""
    # Must specify provider on simulate call
    llm.simulate(
        Provider.OPENAI,
        responses=[
            "Research results",
            "Summary",
            "Recommendations",
        ],
    )

    multi_step_workflow("AI trends")

    # Must specify provider on verify too
    llm.verify_many(Provider.OPENAI, count=3)


def test_with_default_provider_per_test(construct: Construct) -> None:
    """Set default provider at the start of the test."""
    # Set default provider once at the start
    construct.set_default_provider(Provider.OPENAI)

    # Now you can omit the provider argument
    llm.simulate(
        responses=[
            "Research results",
            "Summary",
            "Recommendations",
        ]
    )

    multi_step_workflow("AI trends")

    # Verify also works without provider
    llm.verify_many(count=3)
