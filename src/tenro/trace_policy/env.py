# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Environment variable parsing for trace policy configuration."""

from __future__ import annotations

import os

from tenro.errors.base import TenroError
from tenro.trace_policy.categories import Category
from tenro.trace_policy.modes import CaptureMode

ENV_POLICY = "TENRO_TRACE_POLICY"
ENV_GROUPS = "TENRO_TRACE_CONTENT_GROUPS"

_BOOL_TRUE: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_BOOL_FALSE: frozenset[str] = frozenset({"0", "false", "no", "off"})


class TenroTracePolicyError(TenroError):
    """Umbrella error for trace policy issues."""


class TenroTracePolicyConfigError(TenroTracePolicyError):
    """Raised on malformed env / kwarg configuration of a trace policy."""


def read_capture_from_env() -> CaptureMode | None:
    """Return the ``CaptureMode`` from env, or ``None`` if unset.

    Raises:
        TenroTracePolicyConfigError: If the value is not ``off`` or ``full``.
    """
    raw = os.environ.get(ENV_POLICY)
    if raw is None or not raw.strip():
        return None
    key = raw.strip().lower()
    if key == CaptureMode.CUSTOM.value:
        raise TenroTracePolicyConfigError(
            f"{ENV_POLICY}={key!r} cannot be set via env (requires a transform callback); "
            "use configure_trace_policy(capture=CaptureMode.CUSTOM, transform=...) instead"
        )
    try:
        return CaptureMode(key)
    except ValueError as exc:
        valid = " | ".join(m.value for m in (CaptureMode.OFF, CaptureMode.FULL))
        raise TenroTracePolicyConfigError(
            f"unknown {ENV_POLICY} value {raw!r}; expected {valid}"
        ) from exc


def read_groups_from_env() -> dict[Category, bool]:
    """Parse ``TENRO_TRACE_CONTENT_GROUPS`` into a ``Category → bool`` dict.

    Returns:
        Empty dict if env is unset or empty.

    Raises:
        TenroTracePolicyConfigError: If any entry is malformed.
    """
    raw = os.environ.get(ENV_GROUPS)
    if raw is None or not raw.strip():
        return {}
    out: dict[Category, bool] = {}
    for entry in (e.strip() for e in raw.split(",") if e.strip()):
        category, flag = _parse_one_group(entry)
        out[category] = flag
    return out


def _parse_one_group(entry: str) -> tuple[Category, bool]:
    if "=" not in entry:
        raise TenroTracePolicyConfigError(
            f"malformed group {entry!r} in {ENV_GROUPS}; expected 'group=bool'"
        )
    raw_cat, raw_flag = entry.split("=", 1)
    try:
        category = Category(raw_cat.strip().lower())
    except ValueError as exc:
        known = ", ".join(c.value for c in Category)
        raise TenroTracePolicyConfigError(
            f"unknown group {raw_cat.strip()!r} in {ENV_GROUPS}; known: {known}"
        ) from exc
    flag = _parse_bool(raw_flag.strip().lower(), raw_cat.strip())
    return category, flag


def _parse_bool(raw: str, context: str) -> bool:
    if raw in _BOOL_TRUE:
        return True
    if raw in _BOOL_FALSE:
        return False
    raise TenroTracePolicyConfigError(
        f"bad bool for {context} in {ENV_GROUPS}: {raw!r} "
        f"(expected {'|'.join(sorted(_BOOL_TRUE | _BOOL_FALSE))})"
    )


__all__ = [
    "ENV_GROUPS",
    "ENV_POLICY",
    "TenroTracePolicyConfigError",
    "TenroTracePolicyError",
    "read_capture_from_env",
    "read_groups_from_env",
]
