# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Template loader for fixture-based response generation.

Loads JSON fixtures, validates with Pydantic, and produces dicts for simulation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from tenro._construct.http.registry import ProviderRegistry, UnsupportedProviderError
from tenro._construct.http.schemas.anthropic.messages import Message
from tenro._construct.http.schemas.gemini.generate_content import GenerateContentResponse
from tenro._construct.http.schemas.openai.chat_completions import ChatCompletion


class TemplateLoader:
    """Loads fixtures and validates them with Pydantic templates.

    Flow:
    1. Load JSON fixture from fixtures/{provider}/{route}/{feature}_response.json
    2. Deep merge with user overrides
    3. Validate with Pydantic template (`extra="allow"` for forward compatibility)
    4. Return dict for simulation

    Attributes:
        _templates: Map of (provider, route) pairs to schema classes.

    Examples:
        >>> data = TemplateLoader.load(
        ...     provider="openai",
        ...     route="chat_completions",
        ...     feature="text",
        ...     content="Hello!",
        ...     model="gpt-4-turbo"
        ... )
        >>> assert data["choices"][0]["message"]["content"] == "Hello!"
    """

    # Map (provider, route) → Pydantic template class
    _templates: ClassVar[dict[tuple[str, str], type[Any]]] = {
        ("openai", "chat_completions"): ChatCompletion,
        ("anthropic", "messages"): Message,
        ("gemini", "generate_content"): GenerateContentResponse,
    }

    @classmethod
    def load(
        cls,
        provider: str,
        route: str,
        feature: str = "text",
        **overrides: Any,
    ) -> dict[str, Any]:
        """Load fixture, merge overrides, validate, and return dict.

        Args:
            provider: Provider name (e.g., `openai`, `anthropic`, `gemini`).
            route: API route (e.g., `chat_completions`, `messages`).
            feature: Feature name (e.g., `text`, `tool_calls`).
            **overrides: Fields to override in fixture (content, model, etc.).

        Returns:
            Validated response dict ready for simulation.

        Raises:
            FileNotFoundError: If fixture file doesn't exist.
            ValueError: If provider/route combination not supported.

        Examples:
            >>> data = TemplateLoader.load(
            ...     provider="openai",
            ...     route="chat_completions",
            ...     feature="text",
            ...     content="Custom response"
            ... )
        """
        # Load JSON fixture
        fixture_data = cls._load_fixture(provider, route, feature)

        merged_data = cls._deep_merge(fixture_data, overrides)

        validated = cls._validate(provider, route, merged_data)
        return validated

    @classmethod
    def _load_fixture(cls, provider: str, route: str, feature: str) -> dict[str, Any]:
        """Load JSON fixture from file.

        Args:
            provider: Provider name.
            route: API route.
            feature: Feature name.

        Returns:
            Parsed JSON as dict.

        Raises:
            FileNotFoundError: If fixture doesn't exist.
        """
        package_root = Path(__file__).parent
        fixture_path = package_root / provider / route / f"{feature}_response.json"

        if not fixture_path.exists():
            available_fixtures = cls._list_available_fixtures()
            error_msg = (
                f"Fixture not found: {fixture_path}\nAvailable fixtures:\n{available_fixtures}"
            )
            raise FileNotFoundError(error_msg)

        with open(fixture_path) as f:
            loaded = json.load(f)
            # Ensure it's a dict
            if not isinstance(loaded, dict):
                error_msg = f"Fixture {fixture_path} must be a JSON object, not {type(loaded)}"
                raise ValueError(error_msg)
            return loaded

    @classmethod
    def _deep_merge(cls, base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        """Deep merge overrides into base dict.

        Args:
            base: Base dict (from fixture).
            overrides: Override dict (from user).

        Returns:
            Merged dict (does not modify inputs).

        Examples:
            >>> base = {"a": 1, "b": {"c": 2}}
            >>> overrides = {"b": {"d": 3}, "e": 4}
            >>> result = TemplateLoader._deep_merge(base, overrides)
            >>> assert result == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}
        """
        result = base.copy()

        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge dicts
                result[key] = cls._deep_merge(result[key], value)
            elif (
                key in result
                and isinstance(result[key], list)
                and isinstance(value, list)
                and cls._is_list_of_dicts(result[key])
                and cls._is_list_of_dicts(value)
            ):
                # Merge lists of dicts element-by-element (for choices, content blocks, etc.)
                result[key] = cls._merge_lists(result[key], value)
            else:
                # Direct override
                result[key] = value

        return result

    @classmethod
    def _is_list_of_dicts(cls, items: list[Any]) -> bool:
        """Check if list contains only dicts (for merge eligibility)."""
        return len(items) > 0 and all(isinstance(item, dict) for item in items)

    @classmethod
    def _merge_lists(cls, base: list[Any], overrides: list[Any]) -> list[Any]:
        """Merge two lists element-by-element.

        For each index, if both elements are dicts, deep merge them.
        Otherwise, use the override value.

        Args:
            base: Base list (from fixture).
            overrides: Override list (from user).

        Returns:
            Merged list.
        """
        result = []
        max_len = max(len(base), len(overrides))

        for i in range(max_len):
            if i < len(overrides) and i < len(base):
                base_item = base[i]
                override_item = overrides[i]
                if isinstance(base_item, dict) and isinstance(override_item, dict):
                    result.append(cls._deep_merge(base_item, override_item))
                else:
                    result.append(override_item)
            elif i < len(overrides):
                result.append(overrides[i])
            else:
                # Deep copy to prevent aliasing with base
                item = base[i]
                result.append(cls._deep_merge(item, {}) if isinstance(item, dict) else item)

        return result

    @classmethod
    def _validate(cls, provider: str, route: str, data: dict[str, Any]) -> dict[str, Any]:
        """Validate data with Pydantic template.

        Args:
            provider: Provider name.
            route: API route.
            data: Data to validate.

        Returns:
            Validated dict (from template.model_dump()).

        Raises:
            ValueError: If provider/route not supported.

        Examples:
            >>> data = {"id": "chatcmpl-123", "object": "chat.completion", ...}
            >>> validated = TemplateLoader._validate("openai", "chat_completions", data)
        """
        # First try to get schema from registry
        template_class = cls._get_schema_from_registry(provider, route)

        # Fall back to local templates for backward compatibility
        if template_class is None:
            template_class = cls._templates.get((provider, route))

        if template_class is None:
            available = ", ".join(f"{p}/{r}" for p, r in cls._templates)
            error_msg = f"Unknown provider/route: {provider}/{route}\nAvailable: {available}"
            raise ValueError(error_msg)

        validated_model = template_class(**data)
        result = validated_model.model_dump()
        assert isinstance(result, dict)  # Required for mypy
        return result

    @classmethod
    def _get_schema_from_registry(cls, provider: str, route: str) -> type[Any] | None:
        """Get Pydantic schema from registry.

        Args:
            provider: Provider name.
            route: API route (endpoint name).

        Returns:
            Pydantic schema class or None if not found.
        """
        try:
            family = ProviderRegistry.get_compatibility_family_for_provider(provider)
            if route in family.endpoints:
                return family.endpoints[route].pydantic_schema
            default = family.default_endpoint
            if default.name == route:
                return default.pydantic_schema
        except UnsupportedProviderError:
            pass
        return None

    @classmethod
    def _list_available_fixtures(cls) -> str:
        """List all available fixtures for error messages.

        Returns:
            Formatted string listing all fixtures.
        """
        package_root = Path(__file__).parent
        fixtures_dir = package_root

        if not fixtures_dir.exists():
            return "(test_vectors directory not found)"

        available = []
        for provider_dir in sorted(fixtures_dir.iterdir()):
            if not provider_dir.is_dir():
                continue

            for route_dir in sorted(provider_dir.iterdir()):
                if not route_dir.is_dir():
                    continue

                for fixture_file in sorted(route_dir.glob("*_response.json")):
                    feature = fixture_file.stem.replace("_response", "")
                    available.append(f"  - {provider_dir.name}/{route_dir.name}/{feature}")

        return "\n".join(available) if available else "(no fixtures found)"
