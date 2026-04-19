# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""TracePolicy — the resolved policy governing span content capture."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Final

from tenro.trace_policy.categories import Category
from tenro.trace_policy.modes import CaptureMode, Transform

POLICY_VERSION: Final = "2.1"
"""Wire-format version stamped on every span by the active trace policy."""


class _Unset:
    """Sentinel distinguishing ``None`` from "no value supplied"."""

    _instance: _Unset | None = None

    def __new__(cls) -> _Unset:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final = _Unset()

_ALL_CATEGORIES_ENABLED: Final = MappingProxyType(dict.fromkeys(Category, True))


@dataclass(frozen=True)
class TracePolicy:
    """Resolved trace policy for one process or override scope.

    Attributes:
        capture: Top-level capture mode. ``OFF`` drops content; ``FULL`` emits
            content as-is; ``CUSTOM`` delegates to ``transform``.
        groups: Per-category enable flags. A category whose flag is ``False``
            drops regardless of ``capture``. Missing categories default to
            enabled.
        transform: Callback used when ``capture`` is ``CUSTOM``. Ignored
            otherwise.
        version: Policy schema version. Stamped on every span.
    """

    capture: CaptureMode = CaptureMode.OFF
    groups: Mapping[Category, bool] = field(default_factory=lambda: _ALL_CATEGORIES_ENABLED)
    transform: Transform | None = None
    version: str = POLICY_VERSION

    def __post_init__(self) -> None:
        _validate_capture(self.capture, self.transform)
        normalized = _normalize_groups(self.groups)
        object.__setattr__(self, "groups", normalized)

    def captures(self, category: Category) -> bool:
        """Return ``True`` if ``category`` would emit any content."""
        if self.capture is CaptureMode.OFF:
            return False
        return self.groups.get(category, True)

    def with_overrides(
        self,
        *,
        capture: CaptureMode | _Unset = UNSET,
        transform: Transform | None | _Unset = UNSET,
        groups: Mapping[Category, bool] | None = None,
    ) -> TracePolicy:
        """Return a new policy with selective overrides applied.

        Args:
            capture: Replace the capture mode.
            transform: Replace the transform (``None`` clears it).
            groups: Per-category overrides merged on top of existing flags.

        Returns:
            New ``TracePolicy``. The original is unchanged.
        """
        new_groups = dict(self.groups)
        if groups:
            new_groups.update(groups)
        return replace(
            self,
            capture=self.capture if isinstance(capture, _Unset) else capture,
            transform=self.transform if isinstance(transform, _Unset) else transform,
            groups=new_groups,
        )

    def audit_attributes(self) -> dict[str, Any]:
        """Return the policy attributes stamped on every span."""
        return {
            "tenro.trace.policy": self.capture.value,
            "tenro.trace.policy_version": self.version,
            "tenro.trace.content_mode": self._content_mode(),
        }

    def _content_mode(self) -> str:
        """Return ``none``, ``partial``, ``full``, or ``custom``.

        ``custom`` wins whenever capture is ``CUSTOM`` — a user transform
        can drop or reshape content in ways the group flags cannot describe.
        """
        if self.capture is CaptureMode.CUSTOM:
            return "custom"
        if self.capture is CaptureMode.OFF:
            return "none"
        enabled_count = sum(1 for cat in Category if self.groups.get(cat, True))
        if enabled_count == len(Category):
            return "full"
        if enabled_count == 0:
            return "none"
        return "partial"


def _validate_capture(capture: CaptureMode, transform: Transform | None) -> None:
    if not isinstance(capture, CaptureMode):
        raise TypeError(f"capture must be CaptureMode, got {type(capture).__name__}")
    if capture is CaptureMode.CUSTOM and transform is None:
        raise ValueError("capture=CUSTOM requires a transform callback")
    if capture is not CaptureMode.CUSTOM and transform is not None:
        raise ValueError(
            f"transform is only used when capture=CUSTOM; "
            f"got capture={capture.value!r} with a transform. "
            "Pass capture=CUSTOM or omit the transform — otherwise the "
            "callback would be silently ignored."
        )
    if transform is not None and not callable(transform):
        raise TypeError(f"transform must be callable, got {type(transform).__name__}")


def _normalize_groups(groups: Mapping[Category, bool]) -> Mapping[Category, bool]:
    if not isinstance(groups, Mapping):
        raise TypeError(f"groups must be a Mapping, got {type(groups).__name__}")
    out: dict[Category, bool] = dict.fromkeys(Category, True)
    for key, value in groups.items():
        if not isinstance(key, Category):
            raise TypeError(f"groups keys must be Category, got {type(key).__name__}")
        if not isinstance(value, bool):
            raise TypeError(f"groups[{key.value}] must be bool, got {type(value).__name__}")
        out[key] = value
    return MappingProxyType(out)


__all__ = ["POLICY_VERSION", "UNSET", "TracePolicy"]
