# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Trace visualization and debugging utilities.

Provides tools for visualizing trace output during test runs for debugging
and development purposes.

Enable trace output via:
    - Environment variable: TENRO_TRACE=true
    - Pytest flag: --tenro-trace

Example:
    >>> # Enable via environment
    >>> import os
    >>> os.environ["TENRO_TRACE"] = "true"
    >>>
    >>> # Or check if enabled
    >>> from tenro.trace import is_trace_enabled
    >>> if is_trace_enabled():
    ...     print("Trace output is enabled")
"""

from __future__ import annotations

from tenro.trace.config import TraceConfig, get_trace_config, is_trace_enabled
from tenro.trace.renderer import TraceRenderer

__all__ = [
    "TraceConfig",
    "TraceRenderer",
    "get_trace_config",
    "is_trace_enabled",
]
