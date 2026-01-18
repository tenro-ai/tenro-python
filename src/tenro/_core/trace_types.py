# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Tracing types for Tenro SDK."""

from __future__ import annotations

from typing import Any

# Type aliases for OpenTelemetry
TraceContext = dict[str, Any]
SpanAttributes = dict[str, str | int | float | bool]
