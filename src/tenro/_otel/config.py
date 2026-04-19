# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""OTel export configuration.

Stores config set via ``tenro.init()`` and resolves it with ``TENRO_*``
and ``OTEL_*`` environment variables into a typed ``OtelConfig`` dataclass.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Literal

_DEFAULT_CONTENT_BYTES = 64_000
_MAX_CONTENT_BYTES = 1_000_000
_MIN_CONTENT_BYTES = 1_000

service_name: str | None = None
environment: str | None = None
resource_attributes: dict[str, str] | None = None
otel_diagnostics: Literal["off", "failures", "debug"] | None = None
capture_content: Literal["off", "span"] | None = None
max_content_bytes: int | None = None


@dataclass(frozen=True)
class OtelConfig:
    """Resolved OTel export configuration.

    Created once per Construct from init() args + env vars.
    Immutable after creation.
    """

    service_name: str = "tenro-dev"
    environment: str | None = None
    resource_attributes: dict[str, str] = field(default_factory=dict)
    diagnostics: Literal["off", "failures", "debug"] = "failures"
    capture_content: Literal["off", "span"] = "off"
    max_content_bytes: int = _DEFAULT_CONTENT_BYTES


def _parse_key_value_string(raw: str) -> dict[str, str]:
    """Parse comma-separated key=value pairs.

    Only the first ``=`` splits key from value (values may contain ``=``).
    Empty pairs are skipped.
    """
    result: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip()
        if key:
            result[key] = value
    return result


def _env_or_none(name: str) -> str | None:
    """Read env var, treating empty string as unset."""
    val = os.environ.get(name, "")
    return val if val else None


def resolve_config() -> OtelConfig:
    """Build OtelConfig from init() args + TENRO_* env vars."""
    content_raw = capture_content or _env_or_none("TENRO_CAPTURE_CONTENT") or "off"
    content_mode: Literal["off", "span"] = "span" if content_raw == "span" else "off"

    raw_max = max_content_bytes or _resolve_int_env(
        "TENRO_MAX_CONTENT_BYTES", _DEFAULT_CONTENT_BYTES
    )
    resolved_max_bytes = _clamp_content_bytes(raw_max)

    return OtelConfig(
        service_name=service_name or _env_or_none("OTEL_SERVICE_NAME") or "tenro-dev",
        environment=environment,
        resource_attributes=_resolve_resource_attrs(),
        diagnostics=otel_diagnostics or "failures",
        capture_content=content_mode,
        max_content_bytes=resolved_max_bytes,
    )


def _clamp_content_bytes(value: int) -> int:
    """Clamp max_content_bytes to [1KB, 1MB], warning if capped."""
    from tenro.errors.warnings import TenroConfigWarning, warn

    if value > _MAX_CONTENT_BYTES:
        warn(
            f"max_content_bytes={value} exceeds 1MB cap; clamped to {_MAX_CONTENT_BYTES}",
            TenroConfigWarning,
            stacklevel=4,
        )
        return _MAX_CONTENT_BYTES
    if value < _MIN_CONTENT_BYTES:
        return _MIN_CONTENT_BYTES
    return value


def _resolve_int_env(name: str, default: int) -> int:
    """Read an integer env var with fallback."""
    raw = _env_or_none(name)
    if raw:
        with contextlib.suppress(ValueError):
            return int(raw)
    return default


def _resolve_resource_attrs() -> dict[str, str]:
    """Resolve resource attributes from env vars and init args."""
    res_attrs: dict[str, str] = {}
    env_res = _env_or_none("OTEL_RESOURCE_ATTRIBUTES")
    if env_res:
        res_attrs = _parse_key_value_string(env_res)
    if resource_attributes is not None:
        res_attrs.update(resource_attributes)
    return res_attrs
