# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""PatchEngine for module-level patching at install time.

Patches already-imported target modules in ``sys.modules`` when
``install()`` runs. Patchers currently only mark modules with a sentinel
attribute; real simulation uses transport-level seams (respx) and
``@link_*`` decorators, not import-time interception.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from types import ModuleType

# Marker attribute to track patched modules
_TENRO_PATCHED_MARKER = "_tenro_patched"


@dataclass
class PatcherStatus:
    """Status of a single module patcher."""

    module: str
    late: bool = False  # Module was already imported before patching
    markers_ok: bool = True  # Patch markers attached successfully
    missing: list[str] = field(default_factory=list)  # Attrs expected but not found
    patched: list[str] = field(default_factory=list)  # Attrs patched successfully


@dataclass
class PatchEngineStatus:
    """Aggregate status for the patching engine and all registered patchers."""

    installed: bool = False
    patchers: dict[str, PatcherStatus] = field(default_factory=dict)


class ModulePatcher:
    """Base class for module-specific patchers.

    Each patcher knows how to instrument a specific module (e.g., requests, httpx).
    Patchers are idempotent and mark modules as patched to avoid double-patching.
    """

    # Subclasses set these
    module_name: str = ""

    def patch(self, module: ModuleType) -> None:
        """Apply patches to the module.

        Args:
            module: The module to patch.
        """
        raise NotImplementedError

    def is_patched(self, module: ModuleType) -> bool:
        """Check if module is already patched.

        Args:
            module: The module to check.

        Returns:
            True if module has been patched by Tenro.
        """
        return getattr(module, _TENRO_PATCHED_MARKER, False)

    def mark_patched(self, module: ModuleType) -> None:
        """Mark module as patched.

        Args:
            module: The module to mark.
        """
        setattr(module, _TENRO_PATCHED_MARKER, True)


class PatchEngine:
    """Engine for module-level patching at install time.

    Manages registration of module patchers and patches already-imported
    target modules when ``install()`` runs.
    """

    def __init__(self) -> None:
        """Initialize the patch engine."""
        self._patchers: dict[str, ModulePatcher] = {}
        self._status = PatchEngineStatus()
        # Thread id of the thread currently running install(), or None.
        # Guarded by ``_state_lock``. The lock is only held for short
        # state-machine transitions (enter/exit install, register, uninstall);
        # install steps themselves run without the lock so that patcher code
        # cannot deadlock against waiters.
        self._installing_thread: int | None = None
        self._state_lock = threading.Lock()
        # Snapshot of patchers captured at the start of each install(), used
        # by verify() so it isn't affected by _patchers mutations that happen
        # during patch() callbacks.
        self._install_snapshot: dict[str, ModulePatcher] = {}

    def register(self, patcher: ModulePatcher) -> None:
        """Register a module patcher.

        Must be called before ``install()``; post-install registration raises
        because the new patcher would silently miss both already-imported
        modules and any future imports.

        Args:
            patcher: The patcher to register.

        Raises:
            ValueError: If patcher has no ``module_name``.
            RuntimeError: If called after ``install()``.
        """
        if not patcher.module_name:
            raise ValueError("Patcher must have module_name set")
        # The state lock is never held while patcher code runs, so this
        # cannot deadlock against a patcher synchronously waiting on the
        # caller (e.g. via Thread.join()).
        with self._state_lock:
            if self._status.installed or self._installing_thread is not None:
                raise RuntimeError(
                    f"Cannot register patcher for {patcher.module_name!r} "
                    "after install(); register all patchers before installing "
                    "the engine."
                )
            self._patchers[patcher.module_name] = patcher

    @property
    def targets(self) -> list[str]:
        """Get list of module names being targeted."""
        return list(self._patchers.keys())

    @property
    def status(self) -> PatchEngineStatus:
        """Get the aggregate status of the engine and its patchers."""
        return self._status

    def install(self, strict: bool = False) -> None:
        """Install the patch engine.

        Patches already-imported target modules and records per-patcher status.
        Populates PatchEngineStatus with results.

        Args:
            strict: If True, raise TenroPatchingSetupError for late modules.
                    If False (default), issue a warning instead.

        Raises:
            TenroPatchingSetupError: If strict=True and modules were already imported,
                or if patch markers fail to attach (always).
        """
        # Short-held state lock: check preconditions and claim ownership
        # atomically, then release the lock before running install steps.
        # Holding the lock across patcher code would deadlock if a patcher
        # synchronously waits on a thread that calls register/uninstall.
        tid = threading.get_ident()
        with self._state_lock:
            if self._installing_thread == tid:
                raise RuntimeError(
                    "install() called reentrantly; a patcher's patch() must "
                    "not trigger engine.install()."
                )
            if self._installing_thread is not None:
                raise RuntimeError(
                    "install() is already running on another thread; callers "
                    "must serialize install() externally."
                )
            if self._status.installed:
                return
            self._installing_thread = tid

        try:
            self._patch_imported_modules()
            self.verify()
            self._handle_late_modules(strict=strict)
            # Only mark installed after all steps succeed, so a failed verify()
            # or strict late-module error leaves the engine re-runnable and still
            # accepting new registrations.
            with self._state_lock:
                self._status.installed = True
        finally:
            with self._state_lock:
                self._installing_thread = None

    def _patch_imported_modules(self) -> None:
        """Patch already-imported modules and record per-patcher status.

        Builds the status map in a local dict so concurrent readers of
        ``self._status.patchers`` never observe a partially populated map
        (which would raise "dictionary changed size during iteration").
        The snapshot is installed atomically under the state lock once all
        patching is done.
        """
        # Snapshot the registry: patcher.patch() runs arbitrary callbacks
        # that could mutate _patchers via private access.
        snapshot = list(self._patchers.items())
        new_status: dict[str, PatcherStatus] = {}
        for module_name, patcher in snapshot:
            module = sys.modules.get(module_name)
            is_late = module is not None
            patcher_status = PatcherStatus(module=module_name, late=is_late)

            if module is not None:
                if not patcher.is_patched(module):
                    patcher.patch(module)
                    patcher.mark_patched(module)
                patcher_status.patched = [_TENRO_PATCHED_MARKER]

            new_status[module_name] = patcher_status

        # Publish the snapshot and status map atomically: verify() pairs
        # them to decide success, and must never observe a mix of old
        # status with a new snapshot (or vice versa).
        new_snapshot = dict(snapshot)
        with self._state_lock:
            self._install_snapshot = new_snapshot
            self._status.patchers = new_status

    def _handle_late_modules(self, strict: bool) -> None:
        """Warn (always) and optionally raise for modules imported pre-install."""
        late = self.late_modules
        if not late:
            return

        import warnings

        from tenro.errors.base import TenroLateImportWarning, TenroPatchingSetupError

        warnings.warn(
            f"Tenro loaded late: {late} were imported before Tenro. "
            "Tenro patched them best-effort, but code that already "
            "captured references to those modules (e.g. `from openai import OpenAI` "
            "at module scope) may bypass patching. "
            "Fix: ensure Tenro loads first (pytest plugin/early import). "
            "Safe to ignore if you only use @link_tool/@link_agent decorators, "
            "tenro.register(), or tenro.simulate.llm/tool/agent() — these paths "
            "do not depend on import-time patching.",
            TenroLateImportWarning,
            stacklevel=3,
        )

        if strict:
            raise TenroPatchingSetupError(late_modules=late)

    def uninstall(self) -> None:
        """Uninstall the patch engine.

        Flips the installed flag. Does NOT unpatch already-patched modules.
        Raises if an install() is in flight on another thread rather than
        blocking, since the installing thread may be synchronously waiting
        on the caller.

        Raises:
            RuntimeError: If called while install() is running.
        """
        tid = threading.get_ident()
        with self._state_lock:
            owner = self._installing_thread
            if owner is not None:
                if owner == tid:
                    raise RuntimeError(
                        "uninstall() called reentrantly from install(); "
                        "a patcher's patch() must not trigger uninstall()."
                    )
                raise RuntimeError(
                    "uninstall() called while install() is running on another "
                    "thread; callers must serialize lifecycle calls externally."
                )
            if not self._status.installed:
                return
            self._status.installed = False

    def verify(self) -> None:
        """Verify that patches were successfully applied.

        Checks that all late-patched modules have the patch marker.
        Updates PatcherStatus.markers_ok and .missing on failure.

        Raises:
            TenroPatchingSetupError: If patch markers are missing.
        """
        # Snapshot status map and install snapshot together so a concurrent
        # install() cannot interleave "old status + new snapshot" and cause
        # spurious failures. Only verify() needs this pairing; per-status
        # mutations below operate on live status objects (still safe because
        # PatcherStatus identity is preserved through publication).
        with self._state_lock:
            status_items = list(self._status.patchers.items())
            snapshot = dict(self._install_snapshot)

        missing_markers: list[str] = []

        for module_name, patcher_status in status_items:
            if not patcher_status.late:
                continue  # Only verify late-patched modules

            # Defer to the registered patcher's ``is_patched()`` to respect
            # subclasses with a custom marker; fall back to the sentinel if
            # the snapshot entry is missing (patcher removed itself).
            module = sys.modules.get(module_name)
            if module is None:
                continue
            patcher = snapshot.get(module_name)
            if patcher is not None:
                marker_ok = patcher.is_patched(module)
            else:
                marker_ok = bool(getattr(module, _TENRO_PATCHED_MARKER, False))
            if not marker_ok:
                patcher_status.markers_ok = False
                patcher_status.missing = [_TENRO_PATCHED_MARKER]
                missing_markers.append(module_name)

        if missing_markers:
            from tenro.errors.base import TenroPatchingSetupError

            raise TenroPatchingSetupError(
                late_modules=missing_markers,
                message=(
                    f"Patch markers not found on modules {missing_markers}. "
                    "Patching may have failed."
                ),
            )

    @property
    def is_installed(self) -> bool:
        """Check if the engine is currently installed."""
        return self._status.installed

    @property
    def late_modules(self) -> list[str]:
        """Get list of modules that were already imported at install time."""
        return [name for name, ps in self._status.patchers.items() if ps.late]


# Global singleton instance
_engine: PatchEngine | None = None


def get_engine() -> PatchEngine:
    """Get or create the global PatchEngine instance."""
    global _engine
    if _engine is None:
        _engine = PatchEngine()
    return _engine
