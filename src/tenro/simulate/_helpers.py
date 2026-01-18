# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for simulation facades."""

from __future__ import annotations

from typing import Any

from tenro.errors import TenroSimulationUsageError
from tenro.linking.context import get_active_construct


def get_construct_or_raise() -> Any:
    """Get active Construct or raise TenroSimulationUsageError.

    Returns:
        The active Construct instance.

    Raises:
        TenroSimulationUsageError: If no Construct is active.
    """
    construct = get_active_construct()
    if construct is None:
        raise TenroSimulationUsageError(
            "No active Construct. Use the pytest 'construct' fixture or "
            "enter a Construct context: `with Construct() as c: ...`"
        )
    return construct
