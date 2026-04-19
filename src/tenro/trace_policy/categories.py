# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Content categories governed by trace policy."""

from __future__ import annotations

from enum import StrEnum


class Category(StrEnum):
    """Content groups.

    Attributes:
        INPUTS: Prompts, user messages, agent inputs, retrieval queries.
        OUTPUTS: Model responses, agent outputs.
        TOOL_CONTENT: Tool call arguments and results.
        MEDIA: Image, audio, and video payloads.
        EMBEDDINGS: Vector embedding arrays.
        METADATA: User-attached metadata, invocation params (temperature,
            max_tokens), entity names, identifiers, function signatures,
            caller locations.
    """

    INPUTS = "inputs"
    OUTPUTS = "outputs"
    TOOL_CONTENT = "tool_content"
    MEDIA = "media"
    EMBEDDINGS = "embeddings"
    METADATA = "metadata"


__all__ = ["Category"]
