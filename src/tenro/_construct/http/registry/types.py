# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Type definitions for provider endpoint registry.

Defines the core data structures for describing LLM providers, their API
endpoints, capabilities, and response formats.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class Provider(StrEnum):
    """Built-in LLM providers with type-safe identifiers.

    Use enum members for built-in providers to get IDE autocomplete
    and typo detection. For custom providers, register them first
    with `construct.register_provider()`.

    Examples:
        >>> construct.simulate_llm(Provider.OPENAI, response="Hello!")
        >>> construct.set_default_provider(Provider.ANTHROPIC)
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


_BUILTIN_PROVIDER_VALUES: frozenset[str] = frozenset(p.value for p in Provider)


class Capability(StrEnum):
    """Provider/endpoint capabilities.

    Attributes:
        TEXT: Basic text generation.
        TOOLS: Normalized umbrella for tool/function calling.
        TOOL_CALLS: OpenAI-style tool calling.
        TOOL_USE: Anthropic-style tool use.
        FUNCTION_CALL: Gemini-style function calling.
        STREAMING: Server-sent events streaming.
        JSON_MODE: Force JSON output.
        STRUCTURED_OUTPUT: JSON Schema-based output.
    """

    TEXT = "text"
    TOOLS = "tools"
    TOOL_CALLS = "tool_calls"
    TOOL_USE = "tool_use"
    FUNCTION_CALL = "function_call"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"
    STRUCTURED_OUTPUT = "structured_output"


@dataclass(frozen=True)
class PresetSpec:
    """Metadata for a response preset.

    Attributes:
        name: Preset identifier (e.g., "text", "tool_call", "refusal").
        description: Human-readable description.
        example_kwargs: Example kwargs for documentation.
    """

    name: str
    description: str
    example_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EndpointContract:
    """Contract for a single API endpoint.

    Attributes:
        name: Endpoint identifier (e.g., "chat_completions").
        http_path: HTTP path (e.g., "/v1/chat/completions").
        pydantic_schema: Pydantic model for response validation.
        fixture_base: Relative path to fixtures (e.g., "openai/chat_completions").
        capabilities: Set of supported capabilities.
        presets: Available response presets.
        contract_version: CalVer version (e.g., "2025-12-29").
        deprecated: Whether this endpoint version is deprecated.
        deprecation_message: Optional message explaining deprecation.
    """

    name: str
    http_path: str
    pydantic_schema: type[Any]
    fixture_base: str
    capabilities: frozenset[Capability]
    presets: tuple[PresetSpec, ...] = ()
    contract_version: str = ""
    deprecated: bool = False
    deprecation_message: str | None = None

    def __post_init__(self) -> None:
        """Validate capability consistency."""
        tool_specific = {Capability.TOOL_CALLS, Capability.TOOL_USE, Capability.FUNCTION_CALL}
        caps = self.capabilities

        if caps & tool_specific and Capability.TOOLS not in caps:
            raise ValueError("Tool-specific capability requires Capability.TOOLS")

        if Capability.TOOLS in caps and len(caps & tool_specific) != 1:
            msg = "Capability.TOOLS requires exactly one provider-specific tool capability"
            raise ValueError(msg)


class ResponseTransformer(Protocol):
    """Protocol for response transformation hooks.

    Implementations normalize provider-specific quirks to family-compatible format.
    """

    def __call__(self, response: dict[str, Any]) -> dict[str, Any]:
        """Transform provider response to family-compatible format.

        Args:
            response: Raw provider response dict.

        Returns:
            Normalized response dict.
        """
        ...


@dataclass(frozen=True)
class CompatibilityFamily:
    """Group of providers with compatible API shape.

    Attributes:
        name: Family identifier (e.g., "openai_compatible").
        schema_builder: Class for building responses.
        endpoints: Map of endpoint names to contracts.
        contract_version: CalVer version for the compatibility family.
    """

    name: str
    schema_builder: type[Any]
    endpoints: Mapping[str, EndpointContract]
    contract_version: str

    def __post_init__(self) -> None:
        """Validate endpoint keys match contract names."""
        for key, contract in self.endpoints.items():
            if key != contract.name:
                msg = f"Endpoint key '{key}' must match contract.name '{contract.name}'"
                raise ValueError(msg)

    @property
    def default_endpoint(self) -> EndpointContract:
        """Return the default endpoint.

        Returns the first endpoint in the endpoints mapping. For built-in
        compatibility families, this is the primary endpoint (e.g., chat_completions
        for OpenAI).

        Returns:
            First endpoint in the mapping.
        """
        return next(iter(self.endpoints.values()))


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an LLM provider.

    Attributes:
        name: Provider identifier (e.g., "openai", "mistral").
        base_url: Base API URL for HTTP interception (e.g., "https://api.openai.com").
            Requests to this URL will be intercepted and simulated.
        compatibility_family: Family name for response format (e.g., "openai_compatible").
        detection_patterns: Substrings for auto-detecting provider from function paths.
            Used with @link_llm decorators to match target paths like "mistral.chat".
            Not used for HTTP URL matching (use base_url for that).
        default_target: Default target path for API calls.
        platform: Optional platform identifier (e.g., "aws-bedrock", "azure").
        response_transformer: Optional hook to normalize response quirks.
    """

    name: str
    base_url: str
    compatibility_family: str
    detection_patterns: tuple[str, ...] = ()
    default_target: str | None = None
    platform: str | None = None
    response_transformer: ResponseTransformer | None = None
