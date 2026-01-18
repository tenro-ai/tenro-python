# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Provider response wrapper for LLM provider simulation.

Provides `ProviderResponse`, a flexible wrapper that mimics LLM provider
response objects (OpenAI, Anthropic, Gemini) for testing.
"""

from __future__ import annotations

from typing import Any


class ProviderResponse(dict[str, Any]):
    """Wrapper for LLM provider responses with attribute and dict access.

    Supports both attribute access (response.choices[0]) and dictionary
    access (response["choices"][0]). Compatible with OpenAI, Anthropic,
    Gemini, and custom provider response formats.

    Data keys from the wrapped dictionary are accessible as attributes.

    Examples:
        >>> response = ProviderResponse({
        ...     "choices": [{"message": {"content": "Hello!"}}]
        ... })
        >>> response.choices[0].message.content
        'Hello!'
        >>> response["choices"][0]["message"]["content"]
        'Hello!'
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize with recursive wrapping of nested structures.

        Args:
            data: The dictionary to wrap. Nested dicts and lists are
                automatically wrapped to support attribute access at all levels.
        """
        super().__init__(data)
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = ProviderResponse(value)
            elif isinstance(value, list):
                self[key] = [
                    ProviderResponse(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                self[key] = value

    def __getattribute__(self, name: str) -> Any:
        """Override attribute access to prioritize dict keys over methods.

        This ensures that user data keys (like 'items', 'keys', 'values') take
        precedence over built-in dict methods.

        Args:
            name: The attribute name to access.

        Returns:
            The value associated with the key, or the attribute if not a key.

        Raises:
            AttributeError: If neither key nor attribute exists.
        """
        try:
            _dict = object.__getattribute__(self, "__class__").__bases__[0].__getitem__(self, name)
            return _dict
        except (KeyError, IndexError):
            return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow attribute assignment (obj.attr = value).

        Args:
            name: The attribute name to set.
            value: The value to assign.
        """
        self[name] = value

    def model_dump(self) -> dict[str, Any]:
        """Return plain dictionary representation for Pydantic compatibility.

        Returns:
            Plain dictionary representation.
        """
        return dict(self)

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dictionary.

        Returns:
            Plain dictionary representation.
        """
        return dict(self)
