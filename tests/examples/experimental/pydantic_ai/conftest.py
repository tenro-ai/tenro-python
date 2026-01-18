# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for Pydantic AI examples."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True, scope="module")
def _ensure_event_loop() -> Iterator[None]:
    """Provide a default event loop for sync Pydantic AI examples."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield
    finally:
        loop.close()
        asyncio.set_event_loop(None)
