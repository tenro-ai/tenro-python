# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pytest plugin for Tenro SDK.

This module provides pytest integration for Tenro's agent testing
and evaluation capabilities.

The plugin provides:
- construct: Primary fixture for simulating LLM calls and tools
"""

from __future__ import annotations

from tenro.pytest_plugin.fixtures import construct

__all__ = ["construct"]
