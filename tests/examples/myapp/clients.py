# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""LLM client factories for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anthropic
import openai

if TYPE_CHECKING:
    from google.genai import Client as GeminiClient


def get_openai_client(
    api_key: str = "test-key",
    base_url: str | None = None,
) -> openai.OpenAI:
    """Get an OpenAI client instance.

    Args:
        api_key: API key for authentication.
        base_url: Optional base URL for custom providers (e.g., Mistral).

    Returns:
        Configured OpenAI client.
    """
    if base_url is not None:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    return openai.OpenAI(api_key=api_key)


def get_anthropic_client(
    api_key: str = "test-key",
    max_retries: int | None = None,
) -> anthropic.Anthropic:
    """Get an Anthropic client instance.

    Args:
        api_key: API key for authentication.
        max_retries: Max retry attempts for failed requests.

    Returns:
        Configured Anthropic client.
    """
    if max_retries is not None:
        return anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
    return anthropic.Anthropic(api_key=api_key)


def get_gemini_client(
    api_key: str = "test-key",
    force_httpx_async: bool = False,
) -> GeminiClient:
    """Get a Gemini client instance.

    Args:
        api_key: API key for authentication.
        force_httpx_async: When ``True``, forces the async client to use
            ``httpx.AsyncClient`` instead of the default ``aiohttp`` path
            (needed for tests where ``respx`` interception is required).

    Returns:
        Configured Gemini client.
    """
    import httpx
    from google import genai
    from google.genai.types import HttpOptions

    if force_httpx_async:
        return genai.Client(
            api_key=api_key,
            http_options=HttpOptions(httpx_async_client=httpx.AsyncClient()),
        )
    return genai.Client(api_key=api_key)


def get_async_openai_client(api_key: str = "test-key") -> openai.AsyncOpenAI:
    """Get an async OpenAI client instance.

    Args:
        api_key: API key for authentication.

    Returns:
        Configured async OpenAI client.
    """
    return openai.AsyncOpenAI(api_key=api_key)


def get_async_anthropic_client(api_key: str = "test-key") -> anthropic.AsyncAnthropic:
    """Get an async Anthropic client instance.

    Args:
        api_key: API key for authentication.

    Returns:
        Configured async Anthropic client.
    """
    return anthropic.AsyncAnthropic(api_key=api_key)
