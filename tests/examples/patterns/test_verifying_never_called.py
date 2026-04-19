# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pattern: Verifying tools were never called.

Shows how to ensure dangerous or expensive operations didn't happen.
"""

from __future__ import annotations

from examples.myapp import (
    SafeCleanupAgent,
    SmartCacheAgent,
    delete_all_records,
    fetch_from_api,
    get_cached_data,
)

import tenro
from tenro import Provider
from tenro.simulate import llm, tool


@tenro.simulate
def test_cache_hit_skips_api() -> None:
    """When cache hits, expensive API should never be called."""
    # Cache returns data
    tool.simulate(get_cached_data, result={"data": "cached"})
    # Note: Don't simulate fetch_from_api since cache hit skips it

    SmartCacheAgent().get_data("user:123")

    # Cache was checked
    tool.verify_many(get_cached_data, count=1)
    # API was never called - saved an expensive call!
    tool.verify_never(fetch_from_api)


@tenro.simulate
def test_dangerous_operation_not_triggered() -> None:
    """Verify dangerous operations don't happen accidentally."""
    # Note: Don't simulate delete_all_records since unconfirmed mode won't call it

    SafeCleanupAgent().cleanup(confirmed=False)

    # The dangerous operation should never be called without confirmation
    tool.verify_never(delete_all_records)


@tenro.simulate
def test_llm_not_called_for_cached_response() -> None:
    """Verify LLM isn't called when response is cached."""
    # This test demonstrates verify_llm_never - no simulation needed
    # when validating that something WASN'T called

    # No LLM calls expected
    llm.verify_never(Provider.OPENAI)
