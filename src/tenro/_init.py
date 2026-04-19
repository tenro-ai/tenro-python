# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Top-level SDK initialisation (``tenro.init`` / ``tenro.reset``).

Stores SDK config for lazy consumption by the Construct.
"""

from __future__ import annotations

from typing import Literal

from tenro._otel import config as _otel_config

_initialized: bool = False


def init(
    *,
    service_name: str | None = None,
    environment: str | None = None,
    resource_attributes: dict[str, str] | None = None,
    capture_content: Literal["off", "span"] | None = None,
    max_content_bytes: int | None = None,
) -> None:
    """Configure the Tenro SDK.

    Call once at application startup, before any ``Construct`` is created.

    Args:
        service_name: Logical name for your service. Default: ``"tenro-dev"``.
        environment: Deployment environment (e.g. ``"staging"``,
            ``"production"``).
        resource_attributes: Extra resource attributes attached to spans.
        capture_content: Whether to include LLM message content on spans.
            ``"off"``: no content (default).
            ``"span"``: ``gen_ai.input.messages`` and ``gen_ai.output.messages``
            set as JSON on LLM spans. Overrides ``TENRO_CAPTURE_CONTENT``.
        max_content_bytes: Max bytes per serialized content field.
            Default: 64000. Overrides ``TENRO_MAX_CONTENT_BYTES``.
    """
    global _initialized
    _otel_config.service_name = service_name
    _otel_config.environment = environment
    _otel_config.resource_attributes = (
        dict(resource_attributes) if resource_attributes is not None else None
    )
    _otel_config.capture_content = capture_content
    _otel_config.max_content_bytes = max_content_bytes
    _initialized = True


def reset() -> None:
    """Clear SDK config, restoring the uninitialised state."""
    global _initialized
    _otel_config.service_name = None
    _otel_config.environment = None
    _otel_config.resource_attributes = None
    _otel_config.capture_content = None
    _otel_config.max_content_bytes = None
    _initialized = False


def is_initialized() -> bool:
    """Return whether ``init()`` has been called.

    Returns:
        True if ``init()`` was called and not yet ``reset()``.
    """
    return _initialized
