# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Central registry for LLM providers and compatibility families."""

from __future__ import annotations

from typing import ClassVar

from tenro._construct.http.registry.exceptions import (
    PresetNotFoundError,
    UnsupportedEndpointError,
    UnsupportedProviderError,
)
from tenro._construct.http.registry.types import (
    CompatibilityFamily,
    EndpointContract,
    PresetSpec,
    ProviderConfig,
)
from tenro.errors import TenroDeprecationWarning, warn


class ProviderRegistry:
    """Central registry for LLM providers and their endpoint contracts.

    Example:
        ```python
        from tenro._construct.http.registry import ProviderRegistry

        providers = ProviderRegistry.list_providers()
        # ["openai", "anthropic", "gemini"]

        openai = ProviderRegistry.get_provider("openai")
        print(openai.base_url)  # "https://api.openai.com"
        ```
    """

    _families: ClassVar[dict[str, CompatibilityFamily]] = {}
    _providers: ClassVar[dict[str, ProviderConfig]] = {}

    @classmethod
    def register_compatibility_family(cls, family: CompatibilityFamily) -> None:
        """Register a compatibility family.

        Args:
            family: The family to register.
        """
        cls._families[family.name] = family

    @classmethod
    def register_provider(cls, config: ProviderConfig) -> None:
        """Register a provider config.

        Args:
            config: The provider configuration.
        """
        cls._providers[config.name] = config

    @classmethod
    def reset(cls) -> None:
        """Clear all registered providers and compatibility families.

        Primarily for testing. After reset, call register_builtin_providers()
        to restore default state.
        """
        cls._families.clear()
        cls._providers.clear()

    @classmethod
    def get_provider(cls, name: str) -> ProviderConfig:
        """Get provider by name.

        Args:
            name: Provider name.

        Returns:
            Provider configuration.

        Raises:
            UnsupportedProviderError: If provider not found.

        Example:
            ```python
            provider = ProviderRegistry.get_provider("openai")
            print(provider.base_url)  # "https://api.openai.com"
            ```
        """
        if name not in cls._providers:
            raise UnsupportedProviderError(name, available=list(cls._providers.keys()))
        return cls._providers[name]

    @classmethod
    def get_compatibility_family(cls, name: str) -> CompatibilityFamily:
        """Get compatibility family by name.

        Args:
            name: Family name.

        Returns:
            Compatibility family.

        Raises:
            KeyError: If compatibility family not found.
        """
        if name not in cls._families:
            raise KeyError(f"Family '{name}' not found")
        return cls._families[name]

    @classmethod
    def get_compatibility_family_for_provider(cls, provider: str) -> CompatibilityFamily:
        """Get the compatibility family for a provider.

        Args:
            provider: Provider name.

        Returns:
            Compatibility family the provider belongs to.

        Raises:
            UnsupportedProviderError: If provider not found.
            KeyError: If compatibility family not found.
        """
        config = cls.get_provider(provider)
        return cls.get_compatibility_family(config.compatibility_family)

    @classmethod
    def get_endpoint(cls, provider: str, endpoint: str = "default") -> EndpointContract:
        """Get endpoint contract for a provider.

        Args:
            provider: Provider name.
            endpoint: Endpoint name, or "default" for default endpoint.

        Returns:
            Endpoint contract.

        Raises:
            UnsupportedProviderError: If provider not found.
            UnsupportedEndpointError: If endpoint not found.

        Warning:
            Emits TenroDeprecationWarning if endpoint is deprecated.
        """
        family = cls.get_compatibility_family_for_provider(provider)
        if endpoint == "default":
            contract = family.default_endpoint
        elif endpoint not in family.endpoints:
            raise UnsupportedEndpointError(
                provider, endpoint, available=list(family.endpoints.keys())
            )
        else:
            contract = family.endpoints[endpoint]

        if contract.deprecated:
            message = contract.deprecation_message or (
                f"Contract version '{contract.contract_version}' for '{contract.name}' "
                "is deprecated. Run 'pip install --upgrade tenro' to update."
            )
            warn(message, TenroDeprecationWarning)

        return contract

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered providers.

        Returns:
            List of provider names.
        """
        return list(cls._providers.keys())

    @classmethod
    def list_compatibility_families(cls) -> list[str]:
        """List all registered compatibility families.

        Returns:
            List of compatibility family names.
        """
        return list(cls._families.keys())

    @classmethod
    def list_endpoints(cls, provider: str) -> list[str]:
        """List endpoints for a provider.

        Args:
            provider: Provider name.

        Returns:
            List of endpoint names.

        Raises:
            KeyError: If provider not found.
        """
        family = cls.get_compatibility_family_for_provider(provider)
        return list(family.endpoints.keys())

    @classmethod
    def list_presets(cls, provider: str, endpoint: str = "default") -> list[str]:
        """List presets for a provider endpoint.

        Args:
            provider: Provider name.
            endpoint: Endpoint name, or "default" for default endpoint.

        Returns:
            List of preset names.

        Raises:
            KeyError: If provider or endpoint not found.

        Example:
            ```python
            presets = ProviderRegistry.list_presets("openai")
            # ["text", "tool_call", "refusal"]
            ```
        """
        contract = cls.get_endpoint(provider, endpoint)
        return [preset.name for preset in contract.presets]

    @classmethod
    def get_preset(cls, provider: str, preset: str, endpoint: str = "default") -> PresetSpec:
        """Get a specific preset.

        Args:
            provider: Provider name.
            preset: Preset name.
            endpoint: Endpoint name, or "default" for default endpoint.

        Returns:
            Preset specification.

        Raises:
            UnsupportedProviderError: If provider not found.
            UnsupportedEndpointError: If endpoint not found.
            PresetNotFoundError: If preset not found.
        """
        contract = cls.get_endpoint(provider, endpoint)
        for p in contract.presets:
            if p.name == preset:
                return p
        available = [p.name for p in contract.presets]
        raise PresetNotFoundError(provider, preset, endpoint, available=available)

    @classmethod
    def detect_provider(cls, target: str) -> str | None:
        """Detect provider from target string.

        Matches target against provider detection patterns. Uses first-match-wins
        semantics based on provider registration order.

        Args:
            target: Target path (e.g., "openai.chat.completions.create").

        Returns:
            Detected provider name, or None if not found.

        Note:
            If multiple providers match, returns the first registered match.
            Built-in providers are registered in order: openai, anthropic, gemini.

        Example:
            ```python
            provider = ProviderRegistry.detect_provider("openai.chat.completions")
            # "openai"
            ```
        """
        target_lower = target.lower()
        for name, config in cls._providers.items():
            for pattern in config.detection_patterns:
                if pattern in target_lower:
                    return name
        return None
