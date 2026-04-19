# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for trajectory tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeToolCall:
    """Minimal tool call for testing."""

    display_name: str
    target_path: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)


def calls(*names: str) -> list[FakeToolCall]:
    """Create a list of fake tool calls from names."""
    return [FakeToolCall(display_name=n) for n in names]
