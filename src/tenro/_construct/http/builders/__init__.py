# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Provider schema factory for LLM response and tool call generation.

Create provider-specific responses (OpenAI, Anthropic, Gemini).
Supports custom provider registration for extensibility.

Examples:
    >>> from tenro._construct.http.builders import ProviderSchemaFactory
    >>> # Create response
    >>> response = ProviderSchemaFactory.create_response("openai", "Hello!")
    >>> # Register custom provider
    >>> ProviderSchemaFactory.register("cohere", lambda c, **kw: {"text": c})
    >>> # Reuse existing schema for custom provider
    >>> from tenro._construct.http.builders import OpenAISchema
    >>> ProviderSchemaFactory.register("azure-openai", OpenAISchema.create_response)
"""

from tenro._construct.http.builders.anthropic import AnthropicSchema
from tenro._construct.http.builders.factory import ProviderSchemaFactory
from tenro._construct.http.builders.gemini import GeminiSchema
from tenro._construct.http.builders.openai import OpenAISchema

__all__ = ["AnthropicSchema", "GeminiSchema", "OpenAISchema", "ProviderSchemaFactory"]
