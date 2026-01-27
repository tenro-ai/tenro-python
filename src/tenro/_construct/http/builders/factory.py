# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Provider schema factory for provider-specific response creation.

Creates responses matching OpenAI, Anthropic, Gemini, and custom provider
formats for LLM simulation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from tenro._construct.http.builders.anthropic import AnthropicSchema
from tenro._construct.http.builders.gemini import GeminiSchema
from tenro._construct.http.builders.openai import OpenAISchema
from tenro._construct.http.registry import ProviderRegistry
from tenro._construct.http.registry.exceptions import UnsupportedProviderError
from tenro._core.response_types import ProviderResponse

if TYPE_CHECKING:
    from tenro._construct.simulate.llm.tool_call import ToolCall


class ProviderSchemaFactory:
    """Factory for creating provider-specific responses and tool calls.

    Built-in support: `openai`, `anthropic`, `gemini`, `custom`.
    Extensible via `register_provider()` for custom providers at runtime.

    **Registry Behavior:**
    - Custom providers are registered **globally** (class-level state).
    - All Construct instances share the same custom provider registry.
    - Registration persists for the lifetime of the Python process.

    **Schema Builder Signature:**
    - Builders must accept: `(content: str, **kwargs: Any) -> dict[str, Any]`
    - The **kwargs parameter allows passing metadata (token_usage, tool_calls, etc.)
    - For example:
      `lambda content, **kw: {"text": content, "usage": kw.get("token_usage", {})}`

    Attributes:
        _schemas: Built-in provider schema classes keyed by provider name.
        _custom_schemas: Registry of custom schema builders.
        _detection_patterns: Substrings used for provider auto-detection.
        _default_targets: Default target paths for provider APIs.
        _code_examples: Provider-specific code snippets for error messages.

    Examples:
        >>> from tenro._construct.http.builders import ProviderSchemaFactory
        >>> ProviderSchemaFactory.register(
        ...     "cohere",
        ...     lambda content, **kw: {
        ...         "text": content,
        ...         "usage": kw.get("token_usage", {}),
        ...         "model": kw.get("model", "command-r-plus"),
        ...     },
        ...     ["cohere", "command"]
        ... )
        >>> # Now all Construct instances can use the `cohere` provider
    """

    # Built-in provider schemas (class-level)
    _schemas: ClassVar[dict[str, type[OpenAISchema | AnthropicSchema | GeminiSchema]]] = {
        "openai": OpenAISchema,
        "anthropic": AnthropicSchema,
        "gemini": GeminiSchema,
    }

    # Custom provider schemas (registered at runtime)
    # Schema builders accept (content, **kwargs) for flexibility
    _custom_schemas: ClassVar[dict[str, Callable[..., dict[str, Any]]]] = {}

    # Detection patterns for provider auto-detection
    _detection_patterns: ClassVar[dict[str, list[str]]] = {
        "openai": ["openai"],
        "anthropic": ["anthropic"],
        "gemini": ["gemini", "google.genai"],
    }

    # Default target paths for each provider (used when target not specified)
    _default_targets: ClassVar[dict[str, str]] = {
        "openai": "openai.chat.completions.create",
        "anthropic": "anthropic.resources.messages.Messages.create",
        "gemini": "google.genai.models.Models.generate_content",
    }

    # Code examples for error messages (client instantiation + API call)
    _code_examples: ClassVar[dict[str, dict[str, str]]] = {
        "openai": {
            "client": "client = openai.OpenAI(api_key='test')",
            "call": "response = client.chat.completions.create(...)",
            "extract": "return response.choices[0].message.content",
        },
        "anthropic": {
            "client": "client = anthropic.Anthropic(api_key='test')",
            "call": "response = client.messages.create(...)",
            "extract": "return response.content[0].text",
        },
        "gemini": {
            "client": "client = genai.Client(api_key='test')",
            "call": "response = client.models.generate_content(...)",
            "extract": "return response.text",
        },
    }

    @classmethod
    def register(
        cls,
        name: str,
        schema_builder: Callable[..., dict[str, Any]],
        detection_patterns: list[str] | None = None,
    ) -> None:
        """Register custom provider schema builder.

        Enables testing of any LLM provider (Cohere, Mistral, Ollama, etc.)
        without requiring SDK changes.

        Args:
            name: Provider name (e.g., `cohere`, `ollama`).
            schema_builder: Callable with signature
                `(content: str, **kwargs) -> dict[str, Any]`. The **kwargs
                parameter allows passing optional metadata like `token_usage`,
                `tool_calls`, `model`, etc.
            detection_patterns: Optional list of strings to match in target
                path for auto-detection. Defaults to [`name`].

        Examples:
            >>> from tenro._construct.http.builders import ProviderSchemaFactory
            >>> ProviderSchemaFactory.register(
            ...     "cohere",
            ...     lambda content, **kw: {
            ...         "text": content,
            ...         "model": kw.get("model", "command-r-plus"),
            ...         "usage": kw.get("token_usage", {}),
            ...     },
            ...     ["cohere", "command"],
            ... )
            >>> # Now simulate_llm works with Cohere
            >>> construct.simulate_llm(
            ...     "cohere.chat",
            ...     responses="Hello!",
            ...     provider="cohere",
            ...     token_usage={"input": 10, "output": 5}
            ... )
        """
        cls._custom_schemas[name] = schema_builder

        # Register detection patterns (default to [name])
        patterns = detection_patterns if detection_patterns is not None else [name]
        cls._detection_patterns[name] = patterns

    @classmethod
    def create_response(cls, provider: str, content: str, **kwargs: Any) -> ProviderResponse:
        """Create provider-specific response.

        Args:
            provider: Provider name ("openai", "anthropic", "gemini", "custom",
                or registered).
            content: Response content text.
            **kwargs: Optional metadata to pass to schema builder (token_usage,
                model, etc.).

        Returns:
            Provider-compatible response object supporting both attribute
            and dictionary access patterns.
            For "custom" provider, returns ProviderResponse({"text": content}).

        Raises:
            ValueError: If provider is unknown and not registered.

        Examples:
            >>> # With built-in provider
            >>> response = ProviderSchemaFactory.create_response(
            ...     "openai",
            ...     "Hello!",
            ...     token_usage={"total_tokens": 50},
            ...     model="gpt-4-turbo"
            ... )
            >>> # With custom provider
            >>> ProviderSchemaFactory.register("cohere", lambda c, **kw: {"text": c})
            >>> response = ProviderSchemaFactory.create_response("cohere", "Hi")
        """
        if provider in cls._schemas:
            schema_class = cls._schemas[provider]
            return schema_class.create_response(content, **kwargs)

        if provider in cls._custom_schemas:
            schema_builder = cls._custom_schemas[provider]
            custom_dict = schema_builder(content, **kwargs)
            return ProviderResponse(custom_dict)

        # Fallback to "custom" (generic dict wrapped in ProviderResponse)
        if provider == "custom":
            return ProviderResponse({"text": content})

        # Unknown provider
        all_providers = sorted(list(cls._schemas.keys()) + list(cls._custom_schemas.keys()))
        raise ValueError(
            f"Unknown provider '{provider}'.\n"
            f"Available: {', '.join(all_providers)}, custom\n"
            f"Register custom provider: ProviderSchemaFactory.register(...)"
        )

    @classmethod
    def create_response_from_blocks(
        cls,
        provider: str,
        blocks: list[Any],
        **kwargs: Any,
    ) -> ProviderResponse:
        """Create provider response from ordered blocks.

        Providers that support interleaving (Anthropic, Gemini) preserve block order.
        Providers that don't (OpenAI Chat) flatten: concatenate text, extract tool_calls.

        Args:
            provider: Provider name ("openai", "anthropic", "gemini", or custom).
            blocks: List of str (text) or ToolCall objects in desired order.
            **kwargs: Provider-specific response options.

        Returns:
            Provider-compatible response object.

        Raises:
            ValueError: If provider is unknown.
        """
        if provider in cls._schemas:
            schema_class = cls._schemas[provider]
            return schema_class.create_response_from_blocks(blocks, **kwargs)

        if provider in cls._custom_schemas:
            # Custom providers: flatten to content + tool_calls (legacy behavior)
            from tenro.tool_calls import ToolCall

            text_parts = [b for b in blocks if isinstance(b, str)]
            tool_calls = [b for b in blocks if isinstance(b, ToolCall)]
            content = "".join(text_parts)

            schema_builder = cls._custom_schemas[provider]
            if tool_calls:
                tc_dicts = cls.create_tool_calls(provider, tool_calls)
                custom_dict = schema_builder(content, tool_calls=tc_dicts, **kwargs)
            else:
                custom_dict = schema_builder(content, **kwargs)
            return ProviderResponse(custom_dict)

        all_providers = sorted(list(cls._schemas.keys()) + list(cls._custom_schemas.keys()))
        raise ValueError(
            f"Unknown provider '{provider}'.\n"
            f"Available: {', '.join(all_providers)}, custom\n"
            f"Register custom provider: ProviderSchemaFactory.register(...)"
        )

    @classmethod
    def create_tool_calls(
        cls,
        provider: str,
        tool_calls: list[ToolCall],
        adapter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Create provider-specific tool calls from ToolCall objects.

        Args:
            provider: Provider name ("openai", "anthropic", "gemini", or custom).
            tool_calls: List of normalized ToolCall objects.
            adapter: Optional adapter name for custom providers (e.g., "openai").
                When a custom provider is registered with an adapter, pass the
                adapter here to use its tool call format.

        Returns:
            List of provider-specific tool call dicts.

        Raises:
            ValueError: If provider doesn't support tool calls.

        Examples:
            >>> from tenro import ToolCall
            >>> ProviderSchemaFactory.create_tool_calls("openai", [ToolCall("get_weather")])
            [{'id': 'call_...', 'type': 'function', 'function': {...}}]

            >>> ProviderSchemaFactory.create_tool_calls(
            ...     "anthropic",
            ...     [ToolCall("search", query="AI")]
            ... )
            [{'type': 'tool_use', 'id': 'toolu_...', 'name': 'search', 'input': {...}}]

            >>> # Custom provider with adapter
            >>> ProviderSchemaFactory.create_tool_calls(
            ...     "mistral",
            ...     [ToolCall("search", query="AI")],
            ...     adapter="openai"
            ... )
            [{'id': 'call_...', 'type': 'function', 'function': {...}}]
        """
        tool_dicts: list[str | dict[str, Any]] = [
            {"name": tc.name, "arguments": tc.arguments, "id": tc.call_id} for tc in tool_calls
        ]

        if provider in cls._schemas:
            schema_class = cls._schemas[provider]
            return schema_class.create_tool_calls(tool_dicts)

        # Custom providers: use adapter's schema if provided
        if provider in cls._custom_schemas or provider == "custom":
            if adapter and adapter in cls._schemas:
                schema_class = cls._schemas[adapter]
                return schema_class.create_tool_calls(tool_dicts)
            return []

        raise ValueError(
            f"Unknown provider '{provider}' for tool call conversion.\n"
            f"Built-in providers with tool call support: openai, anthropic, gemini"
        )

    @classmethod
    def detect_provider(cls, target: str) -> str:
        """Detect provider from module path or target name.

        Checks both registry and custom registered providers using
        detection patterns.

        Args:
            target: Module path or target name (e.g., "openai.chat.completions").

        Returns:
            Detected provider name or "custom" if unknown.

        Examples:
            >>> ProviderSchemaFactory.register("cohere", lambda c: {...}, ["cohere"])
            >>> ProviderSchemaFactory.detect_provider("cohere.chat")
            'cohere'
        """
        # First check the registry
        registry_result = ProviderRegistry.detect_provider(target)
        if registry_result is not None:
            return registry_result

        # Fall back to custom detection patterns (for custom providers)
        target_lower = target.lower()
        for provider, patterns in cls._detection_patterns.items():
            # Skip built-in providers (handled by registry)
            if provider in cls._schemas:
                continue
            if any(pattern in target_lower for pattern in patterns):
                return provider

        # Unknown provider - return "custom"
        return "custom"

    @classmethod
    def get_default_target(cls, provider: str) -> str | None:
        """Get default target path for a provider.

        Args:
            provider: Provider name (e.g., "openai", "anthropic").

        Returns:
            Default target path for the provider, or `None` if not registered.

        Examples:
            >>> ProviderSchemaFactory.get_default_target("openai")
            'openai.chat.completions.create'
            >>> ProviderSchemaFactory.get_default_target("custom")
            `None`
        """
        # First check the registry
        try:
            config = ProviderRegistry.get_provider(provider)
            if config.default_target:
                return config.default_target
        except UnsupportedProviderError:
            pass

        # Fall back to local targets (for custom providers)
        return cls._default_targets.get(provider)

    @classmethod
    def get_code_example(cls, provider: str) -> dict[str, str] | None:
        """Get code example for a provider (used in error messages).

        Args:
            provider: Provider name (e.g., "openai", "anthropic").

        Returns:
            Dict with 'client', 'call', 'extract' keys, or `None` if not available.

        Examples:
            >>> ProviderSchemaFactory.get_code_example("openai")
            {'client': 'client = openai.OpenAI(...)', ...}
        """
        return cls._code_examples.get(provider)
