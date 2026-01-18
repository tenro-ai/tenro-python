# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Exception classes for provider registry."""

from __future__ import annotations

from tenro.errors import TenroError


class ProviderError(TenroError):
    """Base exception for provider-related errors."""

    pass


class UnsupportedProviderError(ProviderError):
    """Raised when provider is not in registry.

    Attributes:
        provider: The unsupported provider name.
        available: List of available providers.
    """

    def __init__(self, provider: str, available: list[str] | None = None) -> None:
        """Initialize the error.

        Args:
            provider: The unsupported provider name.
            available: Optional list of available providers.
        """
        self.provider = provider
        self.available = available or []

        if self.available:
            available_str = ", ".join(sorted(self.available))
            message = f"Provider '{provider}' not supported. Available: {available_str}"
        else:
            message = f"Provider '{provider}' not supported."

        super().__init__(message)


class UnsupportedEndpointError(ProviderError):
    """Raised when endpoint is not supported for provider.

    Attributes:
        provider: The provider name.
        endpoint: The unsupported endpoint name.
        available: List of available endpoints.
    """

    def __init__(self, provider: str, endpoint: str, available: list[str] | None = None) -> None:
        """Initialize the error.

        Args:
            provider: The provider name.
            endpoint: The unsupported endpoint name.
            available: Optional list of available endpoints.
        """
        self.provider = provider
        self.endpoint = endpoint
        self.available = available or []

        if self.available:
            available_str = ", ".join(sorted(self.available))
            message = (
                f"Endpoint '{endpoint}' not supported for {provider}. Available: {available_str}"
            )
        else:
            message = f"Endpoint '{endpoint}' not supported for {provider}."

        super().__init__(message)


class PresetNotFoundError(ProviderError):
    """Raised when preset is not found for endpoint.

    Attributes:
        provider: The provider name.
        preset: The preset name.
        endpoint: The endpoint name.
        available: List of available presets.
    """

    def __init__(
        self,
        provider: str,
        preset: str,
        endpoint: str = "default",
        available: list[str] | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            provider: The provider name.
            preset: The preset name.
            endpoint: The endpoint name.
            available: Optional list of available presets.
        """
        self.provider = provider
        self.preset = preset
        self.endpoint = endpoint
        self.available = available or []

        location = f"{provider}/{endpoint}" if endpoint != "default" else provider
        if self.available:
            available_str = ", ".join(sorted(self.available))
            message = f"Preset '{preset}' not found for {location}. Available: {available_str}"
        else:
            message = f"Preset '{preset}' not found for {location}."

        super().__init__(message)


class ContractValidationError(ProviderError):
    """Raised when fixture doesn't match contract schema.

    Attributes:
        provider: The provider name.
        endpoint: The endpoint name.
        errors: List of validation error messages.
    """

    def __init__(self, provider: str, endpoint: str, errors: list[str]) -> None:
        """Initialize the error.

        Args:
            provider: The provider name.
            endpoint: The endpoint name.
            errors: List of validation error messages.
        """
        self.provider = provider
        self.endpoint = endpoint
        self.errors = errors

        error_list = "\n  - ".join(errors)
        message = f"Contract validation failed for {provider}/{endpoint}:\n  - {error_list}"

        super().__init__(message)


# DeprecatedVersionWarning moved to tenro.errors.warnings for centralized warning management
