# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Plugin discovery for community providers via entry points."""

from __future__ import annotations

from importlib.metadata import entry_points

from tenro.errors import TenroPluginWarning, warn


def discover_provider_plugins() -> None:
    """Auto-discover and register provider plugins via entry points.

    Plugins register themselves by defining an entry point in the
    'tenro.providers' group. The entry point should point to a
    callable that, when invoked, registers the provider.

    If a plugin fails to load, a TenroPluginWarning is emitted but
    other plugins continue loading.

    Example plugin pyproject.toml:
        [project.entry-points."tenro.providers"]
        mistral = "tenro_provider_mistral:register"

    Example register function:
        def register():
            from tenro._construct.http.registry import ProviderRegistry, ProviderConfig
            ProviderRegistry.register_provider(ProviderConfig(
                name="mistral",
                base_url="https://api.mistral.ai",
                compatibility_family="openai_compatible",
                detection_patterns=("mistral",),
            ))
    """
    eps = entry_points(group="tenro.providers")
    for ep in eps:
        try:
            register_fn = ep.load()
            register_fn()
        except Exception as e:
            warn(
                f"Failed to load provider plugin '{ep.name}': {e}",
                TenroPluginWarning,
            )
