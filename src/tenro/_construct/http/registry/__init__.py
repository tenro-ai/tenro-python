# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Provider endpoint registry for LLM simulation.

Provides a central registry for provider configurations, compatibility families,
and endpoint contracts. Supports plugin discovery via entry points.
"""

from __future__ import annotations

from tenro._construct.http.registry.builtin import register_builtin_providers
from tenro._construct.http.registry.exceptions import (
    ContractValidationError,
    PresetNotFoundError,
    ProviderError,
    UnsupportedEndpointError,
    UnsupportedProviderError,
)
from tenro._construct.http.registry.plugins import discover_provider_plugins
from tenro._construct.http.registry.registry import ProviderRegistry
from tenro._construct.http.registry.types import (
    Capability,
    CompatibilityFamily,
    EndpointContract,
    PresetSpec,
    ProviderConfig,
    ResponseTransformer,
)

register_builtin_providers()

__all__ = [
    "Capability",
    "CompatibilityFamily",
    "ContractValidationError",
    "EndpointContract",
    "PresetNotFoundError",
    "PresetSpec",
    "ProviderConfig",
    "ProviderError",
    "ProviderRegistry",
    "ResponseTransformer",
    "UnsupportedEndpointError",
    "UnsupportedProviderError",
    "discover_provider_plugins",
    "register_builtin_providers",
]
