# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Payload models for span events.

These models define the structure of the `data` field in SpanEvent,
discriminated by span_kind + event_type.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from tenro._core.model_base import BaseModel


class LLMRequest(BaseModel):
    """Payload for LLM start event.

    Attributes:
        model: Model identifier.
        messages: Message list sent to the provider.
        temperature: Sampling temperature used for the request.
        provider: Provider name (e.g., `openai`).
    """

    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.0
    provider: str = "openai"


class LLMResponse(BaseModel):
    """Payload for LLM end event.

    Attributes:
        text: Model response text.
        usage: Token usage metadata if provided by the provider.
    """

    text: str
    usage: dict[str, int] | None = None


class ToolRequest(BaseModel):
    """Payload for tool start event.

    Attributes:
        tool_name: Tool name.
        args: Positional arguments passed to the tool.
        kwargs: Keyword arguments passed to the tool.
    """

    tool_name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ToolResponse(BaseModel):
    """Payload for tool end event.

    Attributes:
        result: Tool response value.
    """

    result: Any


class AgentInput(BaseModel):
    """Payload for agent start event.

    Attributes:
        agent_name: Agent name.
        input_data: Input payload for the agent.
    """

    agent_name: str
    input_data: dict[str, Any]


class AgentOutput(BaseModel):
    """Payload for agent end event.

    Attributes:
        output_data: Output payload from the agent.
    """

    output_data: Any
