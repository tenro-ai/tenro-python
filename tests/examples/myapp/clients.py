# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""LLM client factories for testing."""

from __future__ import annotations

import anthropic
import openai


def get_openai_client(api_key: str = "test-key") -> openai.OpenAI:
    """Get an OpenAI client instance.

    Args:
        api_key: API key for authentication.

    Returns:
        Configured OpenAI client.
    """
    return openai.OpenAI(api_key=api_key)


def get_anthropic_client(api_key: str = "test-key") -> anthropic.Anthropic:
    """Get an Anthropic client instance.

    Args:
        api_key: API key for authentication.

    Returns:
        Configured Anthropic client.
    """
    return anthropic.Anthropic(api_key=api_key)
