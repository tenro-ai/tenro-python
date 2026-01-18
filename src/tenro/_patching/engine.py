# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""PatchEngine for module-level patching at import time.

Manages sys.modules patching and sys.meta_path import hooks
for third-party module instrumentation during test execution.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any

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
    """Overall patching backend status (single source of truth)."""

    installed: bool = False
    meta_path_installed: bool = False
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


class _TenroImportHook(MetaPathFinder):
    """Import hook that patches modules on import.

    Installed in sys.meta_path to catch imports after PatchEngine.install().
    """

    def __init__(self, patchers: dict[str, ModulePatcher]) -> None:
        """Initialize with registered patchers.

        Args:
            patchers: Map of module_name -> patcher.
        """
        self._patchers = patchers
        self._in_find = False  # Avoid infinite loops during find operations

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        """Find spec hook that observes imports without providing specs.

        Returns None to let the normal import machinery handle the import,
        then patches in the post-import phase via a different mechanism.
        """
        return None

    def patch_if_needed(self, module: ModuleType, fullname: str) -> None:
        """Patch module if a patcher is registered for it.

        Called after module is imported.

        Args:
            module: The imported module.
            fullname: Full module name.
        """
        root = fullname.split(".")[0]
        patcher = self._patchers.get(root)
        if patcher and not patcher.is_patched(module):
            patcher.patch(module)
            patcher.mark_patched(module)


class PatchEngine:
    """Engine for module-level patching at import time.

    Manages registration of module patchers and installation of import hooks
    for third-party module instrumentation during test execution.
    """

    def __init__(self) -> None:
        """Initialize the patch engine."""
        self._patchers: dict[str, ModulePatcher] = {}
        self._import_hook: _TenroImportHook | None = None
        self._status = PatchEngineStatus()
        self._original_import: Callable[..., Any] | None = None

    def register(self, patcher: ModulePatcher) -> None:
        """Register a module patcher.

        Args:
            patcher: The patcher to register.
        """
        if not patcher.module_name:
            raise ValueError("Patcher must have module_name set")
        self._patchers[patcher.module_name] = patcher

    @property
    def targets(self) -> list[str]:
        """Get list of module names being targeted."""
        return list(self._patchers.keys())

    @property
    def status(self) -> PatchEngineStatus:
        """Get the current status (single source of truth)."""
        return self._status

    def install(self, strict: bool = False) -> None:
        """Install the patch engine.

        Patches already-imported modules and installs import hook for future imports.
        Populates PatchEngineStatus with results.

        Args:
            strict: If True, raise TenroPatchingSetupError for late modules.
                    If False (default), issue a warning instead.

        Raises:
            TenroPatchingSetupError: If strict=True and modules were already imported,
                or if patch markers fail to attach (always).
        """
        if self._status.installed:
            return

        # First pass: patch already-imported modules, populate patcher status
        for module_name, patcher in self._patchers.items():
            is_late = module_name in sys.modules
            patcher_status = PatcherStatus(module=module_name, late=is_late)

            if is_late:
                module = sys.modules[module_name]
                if not patcher.is_patched(module):
                    patcher.patch(module)
                    patcher.mark_patched(module)
                # Track what was patched (marker attribute on module)
                patcher_status.patched = [_TENRO_PATCHED_MARKER]

            self._status.patchers[module_name] = patcher_status

        # Install import hook for future imports
        self._import_hook = _TenroImportHook(self._patchers)
        sys.meta_path.insert(0, self._import_hook)
        self._status.meta_path_installed = True

        # Install post-import hook to patch modules after loading
        self._install_import_wrapper()

        self._status.installed = True

        # Verify patches attached (always fail if markers missing - real misconfig)
        self.verify()

        # Collect late modules from status
        late_modules = [name for name, ps in self._status.patchers.items() if ps.late]

        # Warn on late modules (always, even in non-strict mode)
        if late_modules:
            import warnings

            from tenro.errors.base import TenroLateImportWarning

            warnings.warn(
                f"Tenro loaded late: {late_modules} were imported before Tenro. "
                "Tenro patched them best-effort, but some previously-captured "
                "references may bypass patching. "
                "Fix: ensure Tenro loads first (pytest plugin/early import). "
                "Ignore if using @link_*, tenro.register(), or LLM provider simulation.",
                TenroLateImportWarning,
                stacklevel=2,
            )

        # Only raise if strict explicitly enabled
        if late_modules and strict:
            from tenro.errors.base import TenroPatchingSetupError

            raise TenroPatchingSetupError(late_modules=late_modules)

    def _install_import_wrapper(self) -> None:
        """Wrap builtins.__import__ to patch modules post-import."""
        import builtins

        original_import = builtins.__import__
        self._original_import = original_import
        patchers = self._patchers

        def patching_import(
            name: str,
            globals: Any = None,
            locals: Any = None,
            fromlist: Any = (),
            level: int = 0,
        ) -> ModuleType:
            module = original_import(name, globals, locals, fromlist, level)

            # Patch if needed
            root = name.split(".")[0]
            patcher = patchers.get(root)
            if patcher and not patcher.is_patched(module):
                patcher.patch(module)
                patcher.mark_patched(module)

            return module

        builtins.__import__ = patching_import

    def uninstall(self) -> None:
        """Uninstall the patch engine.

        Removes import hook and restores original __import__.
        Does NOT unpatch already-patched modules (remain instrumented).
        """
        if not self._status.installed:
            return

        # Remove import hook
        if self._import_hook and self._import_hook in sys.meta_path:
            sys.meta_path.remove(self._import_hook)
            self._status.meta_path_installed = False

        # Restore original __import__
        if self._original_import:
            import builtins

            builtins.__import__ = self._original_import

        self._status.installed = False

    def verify(self) -> None:
        """Verify that patches were successfully applied.

        Checks that all late-patched modules have the patch marker.
        Updates PatcherStatus.markers_ok and .missing on failure.

        Raises:
            TenroPatchingSetupError: If patch markers are missing.
        """
        missing_markers: list[str] = []

        for module_name, patcher_status in self._status.patchers.items():
            if not patcher_status.late:
                continue  # Only verify late-patched modules

            if module_name in sys.modules:
                module = sys.modules[module_name]
                patcher = self._patchers.get(module_name)
                if patcher and not patcher.is_patched(module):
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
