# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Top-level SDK initialisation (``tenro.init`` / ``tenro.reset``).

Stores OTel config for lazy consumption by the Construct.
"""

from __future__ import annotations

from tenro._otel import config as _otel_config

_initialized: bool = False


def init(
    *,
    resource_attributes: dict[str, str] | None = None,
) -> None:
    """Configure OTel export for Tenro.

    Call once at application startup, before any ``Construct`` is created.

    Args:
        resource_attributes: Extra OTel resource attributes.
    """
    global _initialized
    _otel_config.resource_attributes = (
        dict(resource_attributes) if resource_attributes is not None else None
    )
    _initialized = True


def reset() -> None:
    """Clear OTel config, restoring the uninitialised state."""
    global _initialized
    _otel_config.resource_attributes = None
    _initialized = False


def is_initialized() -> bool:
    """Return whether ``init()`` has been called.

    Returns:
        True if ``init()`` was called and not yet ``reset()``.
    """
    return _initialized
