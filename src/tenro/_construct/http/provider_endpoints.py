# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Provider endpoint discovery for the HTTP interceptor."""

from __future__ import annotations

PROVIDER_ENDPOINTS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
    # Prefix for pattern match.
    "gemini": "https://generativelanguage.googleapis.com/",
}


def get_blocked_domains() -> list[str]:
    """Derive blocked domains from all registered provider endpoints.

    Returns:
        List of domain strings extracted from provider endpoint URLs.
    """
    from urllib.parse import urlparse

    return [urlparse(url).netloc for url in get_provider_endpoints().values()]


def get_provider_endpoints() -> dict[str, str]:
    """Get provider endpoints, including dynamically registered providers.

    Returns:
        Dict mapping provider names to their endpoint URLs.
    """
    from tenro._construct.http.registry import ProviderRegistry

    endpoints = dict(PROVIDER_ENDPOINTS)
    for name in ProviderRegistry.list_providers():
        if name not in endpoints:
            try:
                config = ProviderRegistry.get_provider(name)
                endpoint = ProviderRegistry.get_endpoint(name)
                endpoints[name] = f"{config.base_url}{endpoint.http_path}"
            except Exception:
                pass
    return endpoints


def get_supported_providers() -> set[str]:
    """Get set of providers that support HTTP interception.

    Returns:
        Set of provider names, including dynamically registered providers.
    """
    return set(get_provider_endpoints().keys())


def domain_to_provider(domain: str) -> str | None:
    """Map a domain back to its provider name."""
    for provider, endpoint in get_provider_endpoints().items():
        if domain in endpoint:
            return provider
    return None


def provider_to_domain(provider: str) -> str | None:
    """Map a provider name to its domain."""
    from urllib.parse import urlparse

    endpoint = get_provider_endpoints().get(provider)
    return urlparse(endpoint).netloc if endpoint else None


__all__ = [
    "PROVIDER_ENDPOINTS",
    "domain_to_provider",
    "get_blocked_domains",
    "get_provider_endpoints",
    "get_supported_providers",
    "provider_to_domain",
]
