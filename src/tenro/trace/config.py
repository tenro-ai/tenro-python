# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Configuration for trace visualization.

Handles environment variable parsing and configuration for trace output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_TRUTHY_VALUES = ("true", "1", "yes")


@dataclass(frozen=True)
class TraceConfig:
    """Configuration for trace visualization.

    Attributes:
        enabled: Whether trace output is enabled.
        max_preview_length: Maximum length for input/output previews.
        show_io_preview: Whether to show input/output previews.
    """

    enabled: bool
    max_preview_length: int = 80
    show_io_preview: bool = True


def is_trace_enabled() -> bool:
    """Check if trace output is enabled via TENRO_TRACE env var.

    Returns:
        True if TENRO_TRACE is set to a truthy value (true, 1, yes).

    Example:
        >>> import os
        >>> os.environ["TENRO_TRACE"] = "true"
        >>> is_trace_enabled()
        True
    """
    return os.getenv("TENRO_TRACE", "").lower() in _TRUTHY_VALUES


def get_trace_config() -> TraceConfig:
    """Get trace configuration from environment variables.

    Reads configuration from:
        - TENRO_TRACE: Enable/disable trace output
        - TENRO_TRACE_PREVIEW: Show I/O previews (default: true)
        - TENRO_TRACE_PREVIEW_LENGTH: Max preview length (default: 80)

    Returns:
        TraceConfig with settings from environment.
    """
    preview_env = os.getenv("TENRO_TRACE_PREVIEW", "true").lower()
    length_env = os.getenv("TENRO_TRACE_PREVIEW_LENGTH", "80")

    return TraceConfig(
        enabled=is_trace_enabled(),
        show_io_preview=preview_env in _TRUTHY_VALUES,
        max_preview_length=int(length_env) if length_env.isdigit() else 80,
    )
