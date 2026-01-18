# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Caching utilities for Tenro SDK."""

from __future__ import annotations

import hashlib
from pathlib import Path


def get_cache_dir() -> Path:
    """Get or create cache directory.

    Returns:
        Path to cache directory.
    """
    cache_dir = Path.home() / ".tenro" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def cache_key_hash(key: str) -> str:
    """Generate hash for cache key.

    Args:
        key: Cache key to hash.

    Returns:
        Hashed cache key.
    """
    return hashlib.sha256(key.encode()).hexdigest()
