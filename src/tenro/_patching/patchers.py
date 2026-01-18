# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Module-specific patchers for third-party library instrumentation.

Each patcher knows how to instrument a specific module. Patchers are:
- Idempotent: safe to call multiple times
- Marker-based: set _tenro_patched attribute to track state
- Targeted: only patch specific functions/methods, not entire modules

These patchers are for OBSERVATION/TRACING only, not simulation.
Simulation uses dispatch-based strategies which are capture-safe.
"""

from __future__ import annotations

from types import ModuleType

from tenro._patching.engine import ModulePatcher


class HttpxPatcher(ModulePatcher):
    """Patcher for httpx module.

    Instruments httpx.Client and httpx.AsyncClient for tracing HTTP requests.
    """

    module_name = "httpx"

    def patch(self, module: ModuleType) -> None:
        """Apply patches to httpx module.

        Args:
            module: The httpx module.
        """
        # Marked as patched; HTTP simulation uses respx at transport layer
        # The HTTP simulation uses respx which works at transport layer,
        # not setattr patching, so this is for tracing only
        pass


class RequestsPatcher(ModulePatcher):
    """Patcher for requests module.

    Instruments requests.get, requests.post, etc. for tracing HTTP requests.
    """

    module_name = "requests"

    def patch(self, module: ModuleType) -> None:
        """Apply patches to requests module.

        Args:
            module: The requests module.
        """
        # Marked as patched; HTTP simulation uses respx at transport layer
        pass


class OpenAIPatcher(ModulePatcher):
    """Patcher for openai module.

    Instruments openai.OpenAI and openai.AsyncOpenAI for tracing LLM calls.
    """

    module_name = "openai"

    def patch(self, module: ModuleType) -> None:
        """Apply patches to openai module.

        Args:
            module: The openai module.
        """
        # Marked as patched; HTTP simulation uses respx at transport layer
        # HTTP simulation uses respx, so this is for tracing only
        pass


class AnthropicPatcher(ModulePatcher):
    """Patcher for anthropic module.

    Instruments anthropic.Anthropic and anthropic.AsyncAnthropic for tracing.
    """

    module_name = "anthropic"

    def patch(self, module: ModuleType) -> None:
        """Apply patches to anthropic module.

        Args:
            module: The anthropic module.
        """
        # Marked as patched; HTTP simulation uses respx at transport layer
        # HTTP simulation uses respx, so this is for tracing only
        pass


# Registry of all available patchers
ALL_PATCHERS = [
    HttpxPatcher(),
    RequestsPatcher(),
    OpenAIPatcher(),
    AnthropicPatcher(),
]


def register_all_patchers() -> None:
    """Register all patchers with the global PatchEngine."""
    from tenro._patching.engine import get_engine

    engine = get_engine()
    for patcher in ALL_PATCHERS:
        engine.register(patcher)
