# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Type definitions for simulation module."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from tenro.llm_response import LLMResponse, RawLLMResponse
from tenro.tool_calls import ToolCall

# Type for the responses parameter in simulate_llm
# Accepts scalar forms or list of mixed types
ResponsesInput = (
    str
    | Exception
    | ToolCall
    | LLMResponse
    | RawLLMResponse
    | dict[str, Any]
    | list[str | Exception | ToolCall | LLMResponse | RawLLMResponse | list[Any] | dict[str, Any]]
    | None
)


class SimulationType(StrEnum):
    """Type of simulation target."""

    AGENT = "agent"
    TOOL = "tool"
    LLM = "llm"


class OperationType(StrEnum):
    """Type of operation (simulate or verify)."""

    SIMULATE = "simulate"
    VERIFY = "verify"
