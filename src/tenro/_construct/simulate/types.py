# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Type definitions for simulation module."""

from __future__ import annotations

from enum import StrEnum


class SimulationType(StrEnum):
    """Type of simulation target."""

    AGENT = "agent"
    TOOL = "tool"
    LLM = "llm"


class OperationType(StrEnum):
    """Type of operation (simulate or verify)."""

    SIMULATE = "simulate"
    VERIFY = "verify"
