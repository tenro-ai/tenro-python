# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Provider-specific tool call parsing.

Parses tool call dicts from various provider formats (OpenAI, Anthropic, Gemini)
to a normalized (name, arguments, call_id) tuple.
"""

from __future__ import annotations

import json
from typing import Any


class ToolCallParser:
    """Parse provider-specific tool call formats to (name, arguments, call_id).

    Supports:
    - OpenAI: {"type": "function", "id": ..., "function": {"name": ...,
        "arguments": "..."}}
    - Gemini: {"functionCall": {"name": ..., "args": {...}}}
    - Anthropic: {"type": "tool_use", "id": ..., "name": ..., "input": {...}}
    - Generic: {"name": ..., "arguments"|"input": {...}}
    """

    @staticmethod
    def parse(d: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
        """Detect format and parse to (name, arguments, call_id).

        Args:
            d: Dict in any supported provider format.

        Returns:
            Tuple of (name, arguments dict, call_id or None).

        Raises:
            ValueError: If format is invalid or cannot be parsed.
        """
        if ToolCallParser._is_openai(d):
            return ToolCallParser._parse_openai(d)
        elif ToolCallParser._is_gemini(d):
            return ToolCallParser._parse_gemini(d)
        elif ToolCallParser._is_anthropic(d):
            return ToolCallParser._parse_anthropic(d)
        return ToolCallParser._parse_generic(d)

    # --- Format detection ---

    @staticmethod
    def _is_openai(d: dict[str, Any]) -> bool:
        """Check if dict is OpenAI tool call format."""
        return d.get("type") == "function" and "function" in d

    @staticmethod
    def _is_gemini(d: dict[str, Any]) -> bool:
        """Check if dict is Gemini functionCall format."""
        return "functionCall" in d

    @staticmethod
    def _is_anthropic(d: dict[str, Any]) -> bool:
        """Check if dict is Anthropic tool_use format."""
        return d.get("type") == "tool_use" and "name" in d

    # --- Parsers ---

    @staticmethod
    def _parse_openai(d: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
        """Parse OpenAI format: {"type": "function", "function": {name, arguments}}.

        Note: OpenAI's `arguments` is a JSON string, not a dict.
        """
        func_obj = d["function"]
        if not isinstance(func_obj, dict) or "name" not in func_obj:
            raise ValueError("OpenAI tool call 'function' must have 'name' key")

        name = func_obj["name"]
        args_raw = func_obj.get("arguments", "{}")

        if isinstance(args_raw, str):
            try:
                arguments = json.loads(args_raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"OpenAI tool call 'arguments' is not valid JSON: {e}") from e
        else:
            arguments = args_raw

        return name, arguments, d.get("id")

    @staticmethod
    def _parse_gemini(d: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
        """Parse Gemini format: {"functionCall": {name, args}}.

        Note: Gemini uses "args" (not "arguments") and has no call_id.
        """
        fc = d["functionCall"]
        if not isinstance(fc, dict) or "name" not in fc:
            raise ValueError("Gemini functionCall must have 'name' key")

        name = fc["name"]
        arguments = fc.get("args", {})

        return name, arguments, None  # Gemini has no call_id

    @staticmethod
    def _parse_anthropic(d: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
        """Parse Anthropic format: {"type": "tool_use", name, input, id}.

        Note: Anthropic uses "input" (not "arguments").
        """
        name = d.get("name")
        if not name:
            raise ValueError("Anthropic tool_use must have 'name' key")

        arguments = d.get("input", {})
        call_id = d.get("id")

        return name, arguments, call_id

    @staticmethod
    def _parse_generic(d: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
        """Parse generic format: {"name": ..., "arguments"|"input": {...}}.

        Supports both "arguments" and "input" keys for flexibility.
        """
        if "name" not in d:
            raise ValueError("Dict tool call must have 'name' key")

        name = d["name"]
        arguments = d.get("arguments", d.get("input", {}))
        call_id = d.get("call_id", d.get("id"))

        return name, arguments, call_id


__all__ = ["ToolCallParser"]
