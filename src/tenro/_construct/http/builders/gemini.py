# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Gemini-specific response and tool call schemas."""

from __future__ import annotations

from typing import Any

from tenro._construct.http.builders.base import (
    ToolFormat,
    detect_tool_format,
    extract_tool_name_and_args,
    serialize_arguments_to_dict,
)
from tenro._core.response_types import ProviderResponse


class GeminiSchema:
    """Gemini GenerateContentResponse schema builder.

    Creates minimal responses matching Google's Gemini API format.
    Follows official schema from ai.google.dev/api/rest.
    """

    @staticmethod
    def create_tool_calls(tools: list[str | dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert simplified tools to Gemini tool call format.

        Gemini format: {functionCall: {name, args}}

        Args:
            tools: List in simplified, medium, or full format.

        Returns:
            List of Gemini function call dicts.

        Examples:
            >>> GeminiSchema.create_tool_calls(["get_weather"])
            [{'functionCall': {'name': 'get_weather', 'args': {}}}]
        """
        if not tools:
            return []

        # Detect format using "functionCall" as full format indicator
        tool_format = detect_tool_format(tools, full_format_key="functionCall")

        if tool_format == ToolFormat.SIMPLIFIED:
            return GeminiSchema._convert_simplified(tools)  # type: ignore[arg-type]
        elif tool_format == ToolFormat.MEDIUM:
            return GeminiSchema._convert_medium(tools)  # type: ignore[arg-type]
        else:  # FULL
            return tools  # type: ignore[return-value]

    @staticmethod
    def _convert_simplified(tool_names: list[str]) -> list[dict[str, Any]]:
        """Convert simplified format to Gemini tool calls.

        Args:
            tool_names: List of tool names.

        Returns:
            List of Gemini function call dicts.
        """
        return [
            {"functionCall": {"name": name, "args": {}}}
            for name in tool_names
            if isinstance(name, str)
        ]

    @staticmethod
    def _convert_medium(tool_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert medium format to Gemini tool calls.

        Args:
            tool_dicts: List of dicts with 'name' and 'arguments'.

        Returns:
            List of Gemini function call dicts.
        """
        result = []
        for tool_dict in tool_dicts:
            if not isinstance(tool_dict, dict):
                continue

            name, arguments = extract_tool_name_and_args(tool_dict)
            # Gemini uses "args" (dict) not "arguments"
            args_data = serialize_arguments_to_dict(arguments)

            result.append({"functionCall": {"name": name, "args": args_data}})
        return result

    @staticmethod
    def create_response(content: str, **kwargs: Any) -> ProviderResponse:
        """Create Gemini GenerateContentResponse from fixture.

        Uses TemplateLoader to load JSON fixture, merge overrides, and validate.

        Args:
            content: Response content text.
            **kwargs: Optional response metadata (finish_reason, safety_ratings,
                tool_calls, etc.).
                Supported kwargs:
                - finish_reason: Finish reason (default: from fixture)
                - safety_ratings: List of safety rating dicts
                - tool_calls: List of function call dicts (embedded in parts)

        Returns:
            Gemini-compatible response object supporting both attribute and
            dictionary access (e.g., response.candidates[0] or
            response["candidates"][0]).

        Note:
            Schema matches Google AI API GenerateContentResponse:
            https://ai.google.dev/api/rest/v1beta/GenerateContentResponse
        """
        from tenro._construct.http.test_vectors.loader import TemplateLoader

        # Determine feature based on tool_calls presence
        tool_calls = kwargs.get("tool_calls")
        feature = "function_call" if tool_calls else "text"

        parts: list[dict[str, Any]] = [{"text": content}]
        if tool_calls:
            parts.extend(tool_calls)

        overrides: dict[str, Any] = {
            "candidates": [
                {
                    "content": {
                        "parts": parts,
                    }
                }
            ]
        }

        if "finish_reason" in kwargs:
            overrides["candidates"][0]["finishReason"] = kwargs["finish_reason"]

        if "safety_ratings" in kwargs:
            overrides["candidates"][0]["safetyRatings"] = kwargs["safety_ratings"]

        # Load fixture → validate → return dict
        response_dict = TemplateLoader.load(
            provider="gemini",
            route="generate_content",
            feature=feature,
            **overrides,
        )

        return ProviderResponse(response_dict)
