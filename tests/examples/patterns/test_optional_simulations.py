# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pattern: Optional simulations for conditional branches.

Use `optional=True` when a tool may or may not be called depending on the code
path. Without this flag, Tenro fails if a simulated tool is never invoked.
"""

from __future__ import annotations

from examples.myapp import (
    check_cache,
    fetch_from_api,
    get_data_with_cache,
)

import tenro
from tenro.simulate import tool


@tenro.simulate
def test_cache_hit_skips_api() -> None:
    """When cache returns data, API is not called."""
    tool.simulate(check_cache, result={"data": "cached"})
    tool.simulate(fetch_from_api, result={"data": "fresh"}, optional=True)

    result = get_data_with_cache("key123", use_cache=True)

    assert result == {"data": "cached"}
    tool.verify_many(check_cache, count=1)
    tool.verify_never(fetch_from_api)


@tenro.simulate
def test_cache_miss_calls_api() -> None:
    """When cache returns None, API is called."""
    tool.simulate(check_cache, result=None)
    tool.simulate(fetch_from_api, result={"data": "fresh"})

    result = get_data_with_cache("key123", use_cache=True)

    assert result == {"data": "fresh"}
    tool.verify_many(check_cache, count=1)
    tool.verify_many(fetch_from_api, count=1)


@tenro.simulate
def test_cache_bypass_skips_check() -> None:
    """When cache is disabled, cache check is skipped."""
    tool.simulate(check_cache, result={"data": "cached"}, optional=True)
    tool.simulate(fetch_from_api, result={"data": "fresh"})

    result = get_data_with_cache("key123", use_cache=False)

    assert result == {"data": "fresh"}
    tool.verify_never(check_cache)
    tool.verify_many(fetch_from_api, count=1)
