# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Configuration API for trace policy."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from typing import TypeVar

from tenro.trace_policy.categories import Category
from tenro.trace_policy.env import read_capture_from_env, read_groups_from_env
from tenro.trace_policy.modes import CaptureMode, Transform
from tenro.trace_policy.policy import UNSET, TracePolicy, _Unset

_lock = threading.Lock()
_global_policy: TracePolicy | None = None
_context_stack: ContextVar[tuple[TracePolicy, ...]] = ContextVar(
    "tenro_trace_policy_stack", default=()
)


def configure_trace_policy(
    *,
    capture: CaptureMode | str | _Unset = UNSET,
    transform: Transform | None | _Unset = UNSET,
    groups: Mapping[Category | str, bool] | None = None,
) -> TracePolicy:
    """Install the global trace policy.

    Env vars (``TENRO_TRACE_POLICY`` and ``TENRO_TRACE_CONTENT_GROUPS``) act
    as defaults; explicit kwargs win. ``UNSET`` means "inherit from env or
    the built-in default"; pass ``None`` explicitly to clear an existing
    transform.

    Args:
        capture: Top-level mode. ``"off"`` | ``"full"`` | ``"custom"`` or a
            ``CaptureMode``. When ``"custom"``, ``transform`` is required.
        transform: Callback used when ``capture="custom"``.
        groups: Per-category enable flags. Missing categories stay enabled.

    Returns:
        The resolved ``TracePolicy`` now installed globally.

    Raises:
        TenroTracePolicyConfigError: If env vars are malformed.
        ValueError: If ``capture="custom"`` without a transform.
    """
    policy = TracePolicy(
        capture=_resolve_capture(capture),
        groups=_resolve_groups(groups),
        transform=None if isinstance(transform, _Unset) else transform,
    )
    with _lock:
        global _global_policy
        _global_policy = policy
    return policy


def get_policy() -> TracePolicy:
    """Return the effective trace policy.

    Context-manager overrides via ``override()`` win over the global.
    The global resolves lazily from env on first access.
    """
    stack = _context_stack.get()
    if stack:
        return stack[-1]
    with _lock:
        global _global_policy
        if _global_policy is None:
            _global_policy = _default_from_env()
        return _global_policy


def reset() -> None:
    """Clear the global policy and context stack — test isolation only."""
    with _lock:
        global _global_policy
        _global_policy = None
    _context_stack.set(())


@contextmanager
def override(
    *,
    capture: CaptureMode | str | _Unset = UNSET,
    transform: Transform | None | _Unset = UNSET,
    groups: Mapping[Category | str, bool] | None = None,
) -> Iterator[TracePolicy]:
    """Temporarily override the effective policy, scoped to the current async task or thread.

    Yields:
        The ``TracePolicy`` in effect inside the block.
    """
    new_policy = _resolve_override(capture, transform, groups)
    token = _context_stack.set((*_context_stack.get(), new_policy))
    try:
        yield new_policy
    finally:
        _context_stack.reset(token)


def _default_from_env() -> TracePolicy:
    return TracePolicy(
        capture=read_capture_from_env() or CaptureMode.OFF,
        groups=read_groups_from_env() or {},
    )


def _resolve_override(
    capture: CaptureMode | str | _Unset,
    transform: Transform | None | _Unset,
    groups: Mapping[Category | str, bool] | None,
) -> TracePolicy:
    """Return the resolved policy for an override block.

    Inherits group flags from the base policy when ``groups`` is omitted,
    so outer-scope disables are not silently re-enabled.
    """
    if not isinstance(capture, _Unset) and not isinstance(transform, _Unset) and groups is not None:
        return TracePolicy(
            capture=_coerce_capture(capture),
            transform=transform,
            groups=_coerce_groups(groups),
        )
    base = get_policy()
    return base.with_overrides(
        capture=base.capture if isinstance(capture, _Unset) else _coerce_capture(capture),
        transform=base.transform if isinstance(transform, _Unset) else transform,
        groups=_coerce_groups(groups) if groups else None,
    )


def _resolve_capture(kwarg: CaptureMode | str | _Unset) -> CaptureMode:
    """Explicit kwarg wins; fall back to env only when kwarg is UNSET."""
    if not isinstance(kwarg, _Unset):
        return _coerce_capture(kwarg)
    return read_capture_from_env() or CaptureMode.OFF


def _resolve_groups(
    kwarg_groups: Mapping[Category | str, bool] | None,
) -> dict[Category, bool]:
    """Explicit kwarg wins; fall back to env only when kwarg is None."""
    if kwarg_groups is not None:
        return _coerce_groups(kwarg_groups)
    return dict(read_groups_from_env())


def _coerce_capture(value: CaptureMode | str) -> CaptureMode:
    return _coerce_enum(value, CaptureMode, label="capture mode", valid_prefix="expected")


def _coerce_groups(groups: Mapping[Category | str, bool]) -> dict[Category, bool]:
    return {_coerce_category(key): value for key, value in groups.items()}


def _coerce_category(value: Category | str) -> Category:
    return _coerce_enum(value, Category, label="category", valid_prefix="known:")


_EnumT = TypeVar("_EnumT", bound=StrEnum)


def _coerce_enum(
    value: _EnumT | str,
    enum_cls: type[_EnumT],
    *,
    label: str,
    valid_prefix: str,
) -> _EnumT:
    """Normalize a ``StrEnum``-valued kwarg that may arrive as a string.

    Accepts an enum member or a string; strings are trimmed and lowercased
    so public kwargs mirror env coercion (``"FULL"`` and ``"full"`` both
    resolve). Raises a uniform ``ValueError`` for unknown values listing
    valid choices.
    """
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value.strip().lower())
    except (AttributeError, ValueError) as exc:
        valid = " | ".join(m.value for m in enum_cls)
        raise ValueError(f"unknown {label} {value!r}; {valid_prefix} {valid}") from exc


__all__ = ["configure_trace_policy", "get_policy", "override", "reset"]
