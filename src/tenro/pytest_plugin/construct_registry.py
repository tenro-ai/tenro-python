# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Registry for constructs created during a pytest session."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tenro._construct.construct import Construct

_constructs: list[Construct] = []


def register_construct(construct: Construct) -> None:
    """Register a construct for session-wide export."""
    _constructs.append(construct)


def get_constructs() -> list[Construct]:
    """Get all registered constructs."""
    return list(_constructs)


def clear_constructs() -> None:
    """Clear all registered constructs."""
    _constructs.clear()
