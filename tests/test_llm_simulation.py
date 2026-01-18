# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""LLM simulation tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from myapp.agents import chat_completion

from tenro import Provider
from tenro.simulate import llm

if TYPE_CHECKING:
    import anthropic
    import openai

    from tenro import Construct


@pytest.mark.parametrize(
    ("provider", "client_fixture"),
    [
        pytest.param(Provider.OPENAI, "openai_client", id="openai"),
        pytest.param(Provider.ANTHROPIC, "anthropic_client", id="anthropic"),
    ],
)
class TestLLMSimulation:
    """LLM simulation and verification."""

    def test_single_response(
        self,
        construct: Construct,
        provider: Provider,
        client_fixture: str,
        openai_client: openai.OpenAI,
        anthropic_client: anthropic.Anthropic,
    ) -> None:
        """Simulate a single LLM response."""
        llm.simulate(provider, response="Hello from Tenro!")

        if provider == Provider.OPENAI:
            resp = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
            )
            result = resp.choices[0].message.content
        else:
            resp = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hi"}],
            )
            result = resp.content[0].text

        assert result == "Hello from Tenro!"
        call = llm.verify(provider)
        assert call.simulated is True

    def test_sequential_responses(
        self,
        construct: Construct,
        provider: Provider,
        client_fixture: str,
        openai_client: openai.OpenAI,
        anthropic_client: anthropic.Anthropic,
    ) -> None:
        """Simulate multiple LLM responses in sequence."""
        llm.simulate(provider, responses=["First", "Second", "Third"])

        for expected in ["First", "Second", "Third"]:
            if provider == Provider.OPENAI:
                resp = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Next"}],
                )
                result = resp.choices[0].message.content
            else:
                resp = anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=100,
                    messages=[{"role": "user", "content": "Next"}],
                )
                result = resp.content[0].text
            assert result == expected

        calls = llm.verify_many(provider, count=3)
        assert calls[0].response == "First"
        assert calls[1].response == "Second"
        assert calls[2].response == "Third"
        assert all(c.simulated for c in calls)


class TestLinkedLLM:
    """Tests for @link_llm decorated functions."""

    def test_linked_llm_function(self, construct: Construct) -> None:
        """Simulate LLM response via @link_llm decorated function."""
        llm.simulate(Provider.OPENAI, response="Linked response")

        result = chat_completion("Test prompt")

        assert result == "Linked response"
        call = llm.verify(Provider.OPENAI)
        assert call.simulated is True


class TestCustomProvider:
    """Custom provider registration and simulation."""

    def test_custom_provider_mistral(self, construct: Construct) -> None:
        """Register and simulate a custom provider (Mistral)."""
        import openai

        from tenro._construct.http.registry import ProviderConfig, ProviderRegistry

        # Register Mistral as custom provider (uses OpenAI-compatible format)
        ProviderRegistry.register_provider(
            ProviderConfig(
                name="mistral",
                base_url="https://api.mistral.ai",
                compatibility_family="openai_compatible",
                detection_patterns=("mistral",),
            )
        )

        # Register with construct for simulation routing
        construct.register_provider("mistral", adapter=Provider.OPENAI)

        # Simulate response
        llm.simulate("mistral", response="Hello from Mistral!")

        # Use OpenAI client with Mistral base URL
        client = openai.OpenAI(
            api_key="test-key",
            base_url="https://api.mistral.ai/v1",
        )
        resp = client.chat.completions.create(
            model="mistral-small",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert resp.choices[0].message.content == "Hello from Mistral!"
        call = llm.verify("mistral")
        assert call.simulated is True
