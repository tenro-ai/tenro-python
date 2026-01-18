# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Built-in provider registrations for OpenAI, Anthropic, and Gemini."""

from __future__ import annotations

from tenro._construct.http.builders.anthropic import AnthropicSchema
from tenro._construct.http.builders.gemini import GeminiSchema
from tenro._construct.http.builders.openai import OpenAISchema
from tenro._construct.http.registry.registry import ProviderRegistry
from tenro._construct.http.registry.types import (
    Capability,
    CompatibilityFamily,
    EndpointContract,
    PresetSpec,
    ProviderConfig,
)
from tenro._construct.http.schemas.anthropic.messages import Message
from tenro._construct.http.schemas.gemini.generate_content import GenerateContentResponse
from tenro._construct.http.schemas.openai.chat_completions import ChatCompletion

OPENAI_TEXT_PRESET = PresetSpec(
    name="text",
    description="Standard text response",
    example_kwargs={"finish_reason": "stop"},
)

OPENAI_TOOL_CALL_PRESET = PresetSpec(
    name="tool_call",
    description="Response with tool calls",
    example_kwargs={"finish_reason": "tool_calls"},
)

OPENAI_REFUSAL_PRESET = PresetSpec(
    name="refusal",
    description="Content policy refusal response",
    example_kwargs={"finish_reason": "stop"},
)

ANTHROPIC_TEXT_PRESET = PresetSpec(
    name="text",
    description="Standard text response",
    example_kwargs={"stop_reason": "end_turn"},
)

ANTHROPIC_TOOL_USE_PRESET = PresetSpec(
    name="tool_use",
    description="Response with tool use blocks",
    example_kwargs={"stop_reason": "tool_use"},
)

GEMINI_TEXT_PRESET = PresetSpec(
    name="text",
    description="Standard text response",
    example_kwargs={"finishReason": "STOP"},
)

GEMINI_FUNCTION_CALL_PRESET = PresetSpec(
    name="function_call",
    description="Response with function calls",
    example_kwargs={"finishReason": "STOP"},
)

OPENAI_CHAT_ENDPOINT = EndpointContract(
    name="chat_completions",
    http_path="/v1/chat/completions",
    pydantic_schema=ChatCompletion,
    fixture_base="openai/chat_completions",
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.TOOLS,
            Capability.TOOL_CALLS,
            Capability.STRUCTURED_OUTPUT,
        }
    ),
    presets=(OPENAI_TEXT_PRESET, OPENAI_TOOL_CALL_PRESET, OPENAI_REFUSAL_PRESET),
    contract_version="2025-12-29",
)

OPENAI_FAMILY = CompatibilityFamily(
    name="openai_compatible",
    schema_builder=OpenAISchema,
    endpoints={"chat_completions": OPENAI_CHAT_ENDPOINT},
    contract_version="2025-12-29",
)

OPENAI_PROVIDER = ProviderConfig(
    name="openai",
    base_url="https://api.openai.com",
    compatibility_family="openai_compatible",
    detection_patterns=("openai",),
    default_target="openai.chat.completions.create",
)

ANTHROPIC_MESSAGES_ENDPOINT = EndpointContract(
    name="messages",
    http_path="/v1/messages",
    pydantic_schema=Message,
    fixture_base="anthropic/messages",
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.TOOLS,
            Capability.TOOL_USE,
        }
    ),
    presets=(ANTHROPIC_TEXT_PRESET, ANTHROPIC_TOOL_USE_PRESET),
    contract_version="2025-12-29",
)

ANTHROPIC_FAMILY = CompatibilityFamily(
    name="anthropic_compatible",
    schema_builder=AnthropicSchema,
    endpoints={"messages": ANTHROPIC_MESSAGES_ENDPOINT},
    contract_version="2025-12-29",
)

ANTHROPIC_PROVIDER = ProviderConfig(
    name="anthropic",
    base_url="https://api.anthropic.com",
    compatibility_family="anthropic_compatible",
    detection_patterns=("anthropic",),
    default_target="anthropic.resources.messages.Messages.create",
)

GEMINI_GENERATE_ENDPOINT = EndpointContract(
    name="generate_content",
    http_path="/v1beta/models/{model}:generateContent",
    pydantic_schema=GenerateContentResponse,
    fixture_base="gemini/generate_content",
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.TOOLS,
            Capability.FUNCTION_CALL,
        }
    ),
    presets=(GEMINI_TEXT_PRESET, GEMINI_FUNCTION_CALL_PRESET),
    contract_version="2025-12-29",
)

GEMINI_FAMILY = CompatibilityFamily(
    name="gemini_compatible",
    schema_builder=GeminiSchema,
    endpoints={"generate_content": GEMINI_GENERATE_ENDPOINT},
    contract_version="2025-12-29",
)

GEMINI_PROVIDER = ProviderConfig(
    name="gemini",
    base_url="https://generativelanguage.googleapis.com",
    compatibility_family="gemini_compatible",
    detection_patterns=("gemini", "google.genai"),
    default_target="google.genai.models.Models.generate_content",
)


def register_builtin_providers() -> None:
    """Register all built-in providers and compatibility families.

    Called automatically when the registry module is imported.
    """
    ProviderRegistry.register_compatibility_family(OPENAI_FAMILY)
    ProviderRegistry.register_compatibility_family(ANTHROPIC_FAMILY)
    ProviderRegistry.register_compatibility_family(GEMINI_FAMILY)
    ProviderRegistry.register_provider(OPENAI_PROVIDER)
    ProviderRegistry.register_provider(ANTHROPIC_PROVIDER)
    ProviderRegistry.register_provider(GEMINI_PROVIDER)
