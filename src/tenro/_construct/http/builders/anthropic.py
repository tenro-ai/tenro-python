# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Anthropic-specific response and tool call schemas."""

from __future__ import annotations

import time
from typing import Any

from tenro._construct.http.builders.base import (
    ToolFormat,
    detect_tool_format,
    extract_tool_name_and_args,
    generate_call_id,
    serialize_arguments_to_dict,
)
from tenro._core.response_types import ProviderResponse


class AnthropicSchema:
    """Anthropic Message response schema builder.

    Creates minimal responses matching Anthropic's Messages API format.
    Includes all required fields for SDK compatibility.
    """

    @staticmethod
    def create_tool_calls(tools: list[str | dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert simplified tools to Anthropic tool call format.

        Anthropic format: {type: "tool_use", id, name, input}

        Args:
            tools: List in simplified, medium, or full format.

        Returns:
            List of Anthropic tool use dicts.

        Examples:
            >>> AnthropicSchema.create_tool_calls(["get_weather"])
            [{'type': 'tool_use', 'id': 'toolu_...', 'name': 'get_weather',
              'input': {}}]
        """
        if not tools:
            return []

        # Detect format using "type" == "tool_use" as full format indicator
        tool_format = detect_tool_format(tools, full_format_key="type")

        # Additional check for Anthropic full format
        if tool_format == ToolFormat.FULL:
            first_item = tools[0]
            if isinstance(first_item, dict) and first_item.get("type") == "tool_use":
                return tools  # type: ignore[return-value]
            # Not Anthropic full format, treat as medium
            tool_format = ToolFormat.MEDIUM

        if tool_format == ToolFormat.SIMPLIFIED:
            return AnthropicSchema._convert_simplified(tools)  # type: ignore[arg-type]
        else:  # MEDIUM
            return AnthropicSchema._convert_medium(tools)  # type: ignore[arg-type]

    @staticmethod
    def _convert_simplified(tool_names: list[str]) -> list[dict[str, Any]]:
        """Convert simplified format to Anthropic tool calls.

        Args:
            tool_names: List of tool names.

        Returns:
            List of Anthropic tool use dicts.
        """
        return [
            {
                "type": "tool_use",
                "id": generate_call_id("toolu"),
                "name": name,
                "input": {},
            }
            for name in tool_names
            if isinstance(name, str)
        ]

    @staticmethod
    def _convert_medium(tool_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert medium format to Anthropic tool calls.

        Args:
            tool_dicts: List of dicts with 'name', 'arguments', and optionally 'id'.

        Returns:
            List of Anthropic tool use dicts.
        """
        result = []
        for tool_dict in tool_dicts:
            if not isinstance(tool_dict, dict):
                continue

            name, arguments = extract_tool_name_and_args(tool_dict)
            # Anthropic uses "input" (dict) not "arguments" (JSON string)
            input_data = serialize_arguments_to_dict(arguments)
            call_id = tool_dict.get("id") or generate_call_id("toolu")

            result.append(
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": input_data,
                }
            )
        return result

    @staticmethod
    def create_response(content: str, **kwargs: Any) -> ProviderResponse:
        """Create Anthropic Message response from fixture.

        Uses TemplateLoader to load JSON fixture, merge overrides, and validate.

        Args:
            content: Response content text.
            **kwargs: Optional response metadata (model, token_usage,
                stop_reason, etc.). Supported kwargs:
                - model: Model name (default: from fixture)
                - token_usage: Dict with input_tokens and output_tokens
                - stop_reason: Stop reason (auto-set based on tool_calls)
                - tool_calls: List of tool use dicts (embedded in content)

        Returns:
            Anthropic-compatible response object supporting both attribute
            and dictionary access (e.g., response.content[0] or response["content"][0]).
        """
        from tenro._construct.http.test_vectors.loader import TemplateLoader

        # Determine feature based on tool_calls presence
        tool_calls = kwargs.get("tool_calls")
        feature = "tool_use" if tool_calls else "text"

        if tool_calls and "stop_reason" not in kwargs:
            kwargs["stop_reason"] = "tool_use"

        content_array: list[dict[str, Any]] = [{"type": "text", "text": content}]
        if tool_calls:
            content_array.extend(tool_calls)

        overrides: dict[str, Any] = {
            "content": content_array,
        }

        if "stop_reason" in kwargs:
            overrides["stop_reason"] = kwargs["stop_reason"]

        if "model" in kwargs:
            overrides["model"] = kwargs["model"]

        if "token_usage" in kwargs:
            overrides["usage"] = kwargs["token_usage"]

        overrides["id"] = f"msg_{int(time.time())}"

        # Load fixture → validate → return dict
        response_dict = TemplateLoader.load(
            provider="anthropic",
            route="messages",
            feature=feature,
            **overrides,
        )

        return ProviderResponse(response_dict)
