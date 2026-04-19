# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Registry for constructs created during a pytest session."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tenro._construct.construct import Construct

_constructs: weakref.WeakSet[Construct] = weakref.WeakSet()


def register_construct(construct: Construct) -> None:
    """Register a construct for session-wide export."""
    _constructs.add(construct)


def unregister_construct(construct: Construct) -> None:
    """Unregister a construct (called at fixture teardown)."""
    _constructs.discard(construct)


def get_constructs() -> list[Construct]:
    """Get all currently-alive registered constructs."""
    return list(_constructs)


def clear_constructs() -> None:
    """Clear all registered constructs."""
    _constructs.clear()
