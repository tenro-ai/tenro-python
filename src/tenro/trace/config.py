# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Configuration for trace visualization.

Handles environment variable parsing and configuration for trace output.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

_TRUTHY_VALUES = ("true", "1", "yes")


def _get_env_with_deprecation(new: str, old: str, default: str = "") -> str:
    """Read env var, falling back to deprecated name with a warning."""
    value = os.getenv(new)
    if value is not None:
        return value
    old_value = os.getenv(old)
    if old_value is not None:
        warnings.warn(
            f"{old} is deprecated, use {new} instead",
            DeprecationWarning,
            stacklevel=3,
        )
        return old_value
    return default


@dataclass(frozen=True)
class TraceConfig:
    """Configuration for trace visualization.

    Attributes:
        enabled: Whether trace output is enabled.
        max_preview_length: Maximum length for input/output previews.
        show_simulation_marker: Whether to show [SIM] marker on simulated spans.
    """

    enabled: bool
    max_preview_length: int = 80
    show_simulation_marker: bool = True


def is_trace_enabled() -> bool:
    """Check if trace output is enabled via TENRO_PRINT_TRACE env var.

    Falls back to deprecated TENRO_TRACE with a warning.

    Returns:
        True if TENRO_PRINT_TRACE is set to a truthy value (true, 1, yes).

    Example:
        >>> import os
        >>> os.environ["TENRO_PRINT_TRACE"] = "true"
        >>> is_trace_enabled()
        True
    """
    return _get_env_with_deprecation("TENRO_PRINT_TRACE", "TENRO_TRACE").lower() in _TRUTHY_VALUES


def get_trace_config() -> TraceConfig:
    """Get trace configuration from environment variables.

    Reads configuration from:
        - TENRO_PRINT_TRACE: Enable/disable trace output
        - TENRO_PRINT_TRACE_LENGTH: Max I/O preview length (default: 80)
        - TENRO_PRINT_TRACE_SIM_MARKER: Show [SIM] marker on simulated spans (default: true)

    Deprecated aliases:
        - TENRO_TRACE → TENRO_PRINT_TRACE
        - TENRO_TRACE_SIM_MARKER → TENRO_PRINT_TRACE_SIM_MARKER
        - TENRO_TRACE_PREVIEW_LENGTH → TENRO_PRINT_TRACE_LENGTH

    Returns:
        TraceConfig with settings from environment.
    """
    length_env = _get_env_with_deprecation(
        "TENRO_PRINT_TRACE_LENGTH", "TENRO_TRACE_PREVIEW_LENGTH", "80"
    )
    sim_marker_env = _get_env_with_deprecation(
        "TENRO_PRINT_TRACE_SIM_MARKER", "TENRO_TRACE_SIM_MARKER", "true"
    ).lower()

    return TraceConfig(
        enabled=is_trace_enabled(),
        max_preview_length=int(length_env) if length_env.isdigit() else 80,
        show_simulation_marker=sim_marker_env in _TRUTHY_VALUES,
    )
