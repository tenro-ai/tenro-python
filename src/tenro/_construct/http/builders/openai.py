# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-specific response and tool call schemas."""

from __future__ import annotations

import time
from typing import Any

from tenro._construct.http.builders.base import (
    ToolFormat,
    detect_tool_format,
    extract_tool_name_and_args,
    generate_call_id,
    serialize_arguments_to_json,
)
from tenro._core.response_types import ProviderResponse


class OpenAISchema:
    """OpenAI ChatCompletion response schema builder.

    Creates minimal responses matching OpenAI's chat completion format.
    Includes all required fields for SDK compatibility.
    """

    @staticmethod
    def create_tool_calls(tools: list[str | dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert simplified tools to OpenAI tool call format.

        OpenAI format: {id, type, function: {name, arguments}}

        Args:
            tools: List in simplified, medium, or full format.

        Returns:
            List of OpenAI tool call dicts.

        Examples:
            >>> OpenAISchema.create_tool_calls(["get_weather"])
            [{'id': 'call_...', 'type': 'function', 'function': {...}}]
        """
        if not tools:
            return []

        # Detect format using "function" as full format indicator
        tool_format = detect_tool_format(tools, full_format_key="function")

        if tool_format == ToolFormat.SIMPLIFIED:
            return OpenAISchema._convert_simplified(tools)  # type: ignore[arg-type]
        elif tool_format == ToolFormat.MEDIUM:
            return OpenAISchema._convert_medium(tools)  # type: ignore[arg-type]
        else:  # FULL
            return tools  # type: ignore[return-value]

    @staticmethod
    def _convert_simplified(tool_names: list[str]) -> list[dict[str, Any]]:
        """Convert simplified format to OpenAI tool calls.

        Args:
            tool_names: List of tool names.

        Returns:
            List of OpenAI tool call dicts.
        """
        return [
            {
                "id": generate_call_id("call"),
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            for name in tool_names
            if isinstance(name, str)
        ]

    @staticmethod
    def _convert_medium(tool_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert medium format to OpenAI tool calls.

        Args:
            tool_dicts: List of dicts with 'name', 'arguments', and optionally 'id'.

        Returns:
            List of OpenAI tool call dicts.
        """
        result = []
        for tool_dict in tool_dicts:
            if not isinstance(tool_dict, dict):
                continue

            name, arguments = extract_tool_name_and_args(tool_dict)
            args_str = serialize_arguments_to_json(arguments)
            call_id = tool_dict.get("id") or generate_call_id("call")

            result.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }
            )
        return result

    @staticmethod
    def create_response(content: str, **kwargs: Any) -> ProviderResponse:
        """Create OpenAI ChatCompletion response from fixture.

        Uses TemplateLoader to load JSON fixture, merge overrides, and validate.

        Args:
            content: Response content text.
            **kwargs: Optional response metadata (model, token_usage, tool_calls, etc.).
                Supported kwargs:
                - model: Model name (default: from fixture)
                - token_usage: Dict with prompt_tokens, completion_tokens, total_tokens
                - finish_reason: Completion finish reason (auto-set based on tool_calls)
                - tool_calls: List of tool call dicts for function calling

        Returns:
            OpenAI-compatible response object supporting both attribute
            and dictionary access (e.g., response.choices[0] or response["choices"][0]).
        """
        from tenro._construct.http.test_vectors.loader import TemplateLoader

        # Determine feature based on tool_calls presence
        tool_calls = kwargs.get("tool_calls")
        feature = "tool_calls" if tool_calls else "text"

        if tool_calls and "finish_reason" not in kwargs:
            kwargs["finish_reason"] = "tool_calls"

        overrides: dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "content": content,
                    }
                }
            ]
        }

        if tool_calls:
            overrides["choices"][0]["message"]["tool_calls"] = tool_calls

        if "finish_reason" in kwargs:
            overrides["choices"][0]["finish_reason"] = kwargs["finish_reason"]

        if "model" in kwargs:
            overrides["model"] = kwargs["model"]

        if "token_usage" in kwargs:
            overrides["usage"] = kwargs["token_usage"]

        overrides["id"] = f"chatcmpl-{int(time.time())}"
        overrides["created"] = int(time.time())

        # Load fixture → validate → return dict
        response_dict = TemplateLoader.load(
            provider="openai",
            route="chat_completions",
            feature=feature,
            **overrides,
        )

        return ProviderResponse(response_dict)

    @staticmethod
    def create_response_from_blocks(blocks: list[Any], **kwargs: Any) -> ProviderResponse:
        """Create OpenAI response from blocks (flattening - no interleaving).

        OpenAI Chat API does not support interleaving. Text blocks are concatenated,
        tool calls are extracted to separate tool_calls array.

        Args:
            blocks: List of str (text) or ToolCall objects.
            **kwargs: Optional response metadata (model, token_usage, finish_reason).

        Returns:
            OpenAI-compatible response with flattened content + tool_calls.
        """
        from tenro._construct.http.builders.base import iterate_blocks

        text_parts, tc_objects = iterate_blocks(blocks)
        content = "".join(text_parts)

        tool_calls = [
            {
                "id": tc.call_id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": serialize_arguments_to_json(tc.arguments),
                },
            }
            for tc in tc_objects
        ]

        if tool_calls:
            return OpenAISchema.create_response(content, tool_calls=tool_calls, **kwargs)
        else:
            return OpenAISchema.create_response(content, **kwargs)
