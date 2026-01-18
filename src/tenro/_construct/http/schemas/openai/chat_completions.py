# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Chat Completions API Pydantic template.

Route: POST /v1/chat/completions
Docs: https://platform.openai.com/docs/api-reference/chat
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatCompletion(BaseModel):
    """OpenAI ChatCompletion response schema.

    Forward-compatible template that validates core fields while
    allowing new fields added by OpenAI (reasoning_effort, system_fingerprint, etc.).

    Schema from: https://platform.openai.com/docs/api-reference/chat/object

    Attributes:
        id: Completion identifier.
        object: Object type for the response.
        created: Unix timestamp when the completion was created.
        model: Model identifier.
        choices: List of completion choices.
        usage: Token usage metadata.
    """

    id: str
    object: str
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any]

    # Forward compatibility: Allow new fields from future API versions
    # Examples: reasoning_effort, system_fingerprint, service_tier
    model_config = ConfigDict(extra="allow")
