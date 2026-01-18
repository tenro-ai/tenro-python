# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pattern: Optional simulations for conditional branches.

Shows how to test code paths that may or may not be taken.
This pattern is useful for testing caching, feature flags, and error paths.

Note: These tests demonstrate Tenro's tool simulation features for conditional
code paths. LLM is not used because the pattern being tested is about
tool invocation based on application logic, not LLM decisions.
"""

from examples.myapp import (
    check_cache,
    fetch_from_api,
    get_data_with_cache,
)

from tenro.simulate import tool


def test_cache_hit_skips_api(construct) -> None:
    """When cache returns data, API is not called."""
    tool.simulate(check_cache, result={"data": "cached"})
    # Note: Don't simulate fetch_from_api - it won't be called

    result = get_data_with_cache("key123", use_cache=True)

    assert result == {"data": "cached"}
    tool.verify_many(check_cache, count=1)
    tool.verify_never(fetch_from_api)


def test_cache_miss_calls_api(construct) -> None:
    """When cache returns None, API is called."""
    tool.simulate(check_cache, result=None)
    tool.simulate(fetch_from_api, result={"data": "fresh"})

    result = get_data_with_cache("key123", use_cache=True)

    assert result == {"data": "fresh"}
    tool.verify_many(check_cache, count=1)
    tool.verify_many(fetch_from_api, count=1)


def test_cache_bypass_skips_check(construct) -> None:
    """When cache is disabled, cache check is skipped."""
    tool.simulate(fetch_from_api, result={"data": "fresh"})
    # Note: Don't simulate check_cache - it won't be called

    result = get_data_with_cache("key123", use_cache=False)

    assert result == {"data": "fresh"}
    tool.verify_never(check_cache)
    tool.verify_many(fetch_from_api, count=1)
