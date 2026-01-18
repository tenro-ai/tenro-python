# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Anthropic Messages API Pydantic template.

Route: POST /v1/messages
Docs: https://docs.anthropic.com/en/api/messages
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    """Anthropic Message response schema.

    Forward-compatible template that validates core fields while
    allowing new fields added by Anthropic.

    Schema from: https://docs.anthropic.com/en/api/messages

    Attributes:
        id: Message identifier.
        type: Message object type.
        role: Message role.
        content: Content blocks for the message.
        model: Model identifier.
        stop_reason: Stop reason if the response ended early.
        usage: Token usage metadata.
    """

    id: str
    type: str
    role: str
    content: list[dict[str, Any]]
    model: str
    stop_reason: str | None
    usage: dict[str, Any]

    # Forward compatibility: Allow new fields from future API versions
    model_config = ConfigDict(extra="allow")
