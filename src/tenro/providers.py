# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Provider enum for LLM providers."""

from __future__ import annotations

from enum import StrEnum


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


__all__ = ["Provider"]
