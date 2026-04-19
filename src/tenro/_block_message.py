# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Shared resolution-guidance text for unsimulated LLM calls."""

from __future__ import annotations

_DOMAIN_TO_PROVIDER: dict[str, str] = {
    "api.openai.com": "Provider.OPENAI",
    "api.anthropic.com": "Provider.ANTHROPIC",
    "generativelanguage.googleapis.com": "Provider.GEMINI",
}


def suggested_provider_arg(domain: str) -> str:
    """Return the ``Provider`` literal to suggest for ``domain``.

    Args:
        domain: Blocked LLM provider domain (e.g. ``api.openai.com``).

    Returns:
        ``Provider.X`` literal for known domains; ``provider=...``
        placeholder for custom providers.
    """
    return _DOMAIN_TO_PROVIDER.get(domain, "provider=...")


def resolution_guidance(domain: str) -> str:
    """Return the user-facing fix instructions for ``domain``.

    Args:
        domain: Blocked LLM provider domain.

    Returns:
        Multi-line fix text suitable for an exception message or pytest
        report section. No leading or trailing newlines.
    """
    provider_arg = suggested_provider_arg(domain)
    return (
        "Fix this by either:\n\n"
        "  1. Faking the response in your test:\n\n"
        "       from tenro.simulate import llm\n"
        f'       llm.simulate({provider_arg}, response="hi")\n\n'
        "  2. Letting the real API run (costs money):\n\n"
        "       @tenro.simulate(allow_real_llm_calls=True)"
    )
