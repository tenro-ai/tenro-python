# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Module patching for third-party library instrumentation.

Automatically patches third-party libraries (requests, httpx, openai, anthropic)
for simulation and tracing integration.

For your own code, use @link_tool/@link_agent decorators or tenro.register()
for capture-safe simulation.
"""

from __future__ import annotations

from tenro._patching.engine import PatchEngine, PatchEngineStatus, PatcherStatus

__all__ = ["PatchEngine", "PatchEngineStatus", "PatcherStatus"]
