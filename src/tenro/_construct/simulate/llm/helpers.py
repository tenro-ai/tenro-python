# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for LLM simulation logic.

Extracts validation, normalization, and decision logic from simulate_llm().
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def validate_llm_simulation_params(
    response: str | None,
    responses: str | Exception | list[str | Exception] | None,
) -> None:
    """Validate mutually exclusive LLM simulation parameters.

    Args:
        response: Single string response.
        responses: Single or list of responses (strings or exceptions).

    Raises:
        ValueError: If both or neither are provided.
    """
    if response is not None and responses is not None:
        raise ValueError("Only one of 'response' or 'responses' can be provided")

    if response is None and responses is None:
        raise ValueError("Either 'response' or 'responses' must be provided")


def resolve_provider_from_target(
    target: str | Callable[..., Any] | None,
    provider: str | None,
    detect_provider_fn: Callable[[str], str],
) -> str:
    """Resolve provider from target if not explicitly provided.

    Args:
        target: Target path or callable.
        provider: Explicit provider name.
        detect_provider_fn: Function to detect provider from path.

    Returns:
        Resolved provider name.

    Raises:
        ValueError: If neither target nor provider is specified.
    """
    if provider is not None:
        return provider

    if target is None:
        raise ValueError(
            "Either 'target' or 'provider' must be specified.\n"
            "Examples:\n"
            "  - construct.simulate_llm(Provider.OPENAI, response='Hi')\n"
            "  - construct.simulate_llm(\n"
            "        target='openai.chat.completions.create', response='Hi'\n"
            "    )"
        )

    if callable(target) and not isinstance(target, str):
        qualname = getattr(target, "__qualname__", "")
        module = getattr(target, "__module__", "")
        temp_path = f"{module}.{qualname}" if module and qualname else str(target)
    else:
        temp_path = str(target)

    return detect_provider_fn(temp_path)


def should_use_http_interception(
    use_http: bool | None,
    custom_target_provided: bool,
    provider: str,
    supported_providers: set[str],
) -> bool:
    """Determine whether to use HTTP interception vs setattr patching.

    Args:
        use_http: Explicit user preference (None = auto-detect).
        custom_target_provided: Whether a custom target was specified.
        provider: The resolved provider name.
        supported_providers: Set of providers with HTTP endpoint support.

    Returns:
        True if HTTP interception should be used, False for setattr.
    """
    if use_http is not None:
        return use_http

    # Auto-detect: use HTTP when provider-only mode with known provider
    return not custom_target_provided and provider in supported_providers
