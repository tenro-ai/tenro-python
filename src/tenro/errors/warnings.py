# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Warning classes for Tenro SDK.

Warning hierarchy:
- TenroWarning (UserWarning) - Base for most SDK warnings
- TenroDeprecationWarning (DeprecationWarning) - API will be removed
- TenroFutureWarning (FutureWarning) - Behavior will change

Example:
    Filter SDK warnings::

        import warnings
        warnings.filterwarnings("ignore", category=TenroWarning)
        warnings.filterwarnings("ignore", category=TenroDeprecationWarning)
"""

from __future__ import annotations

import warnings as _warnings

# No offset needed - stacklevel passed by caller is correct as-is
# With stacklevel=2 (default), the warning points to the caller of warn()


class TenroWarning(UserWarning):
    """Base warning for all Tenro SDK warnings.

    Use this to filter all SDK warnings at once.

    Example:
        >>> import warnings
        >>> warnings.filterwarnings("ignore", category=TenroWarning)
    """


class TenroDeprecationWarning(TenroWarning, DeprecationWarning):
    """Emitted when using a deprecated API that will be removed.

    Inherits from both TenroWarning (for SDK filtering) and DeprecationWarning
    (for standard Python filtering behavior).
    """


class TenroFutureWarning(TenroWarning, FutureWarning):
    """Emitted when behavior will change in an upcoming release.

    Inherits from both TenroWarning (for SDK filtering) and FutureWarning
    (for standard Python filtering behavior).
    """


class TenroPluginWarning(TenroWarning):
    """Emitted when a provider plugin fails to load.

    Other plugins continue loading; only the failing plugin is skipped.
    """


class TenroConfigWarning(TenroWarning):
    """Emitted for non-fatal configuration issues.

    The SDK continues operating but behavior may not match expectations.
    """


class TenroTracingWarning(TenroWarning):
    """Emitted when tracing cannot be applied to a linked component.

    The component still functions but without extended tracing.
    """


class TenroCoercionWarning(TenroWarning):
    """Emitted when ToolCall() automatically converts a value to JSON-serializable format.

    This helps catch unexpected coercions. If you expect the coercion, silence with:
        warnings.filterwarnings("ignore", category=TenroCoercionWarning)
    """


class TenroUnusedSimulationWarning(TenroWarning):
    """Emitted when a simulation was registered but never triggered.

    The simulation was set up but the code path that would trigger it
    was never executed. Use `optional=True` to suppress this warning
    for intentionally optional simulations.
    """


class TenroPatchingWarning(TenroWarning):
    """Emitted when PatchEngine fails to install in non-strict mode.

    The SDK continues operating but simulation may not work correctly
    for captured function references. Use --tenro-strict-patch to
    make patching failures fatal.
    """


class TenroLateImportWarning(TenroWarning):
    """Emitted when modules are imported before Tenro can patch them.

    Best-effort patching is applied but stale references may exist.
    Import tenro before other libraries to avoid this warning.
    """


def warn(
    message: str,
    category: type[Warning] = TenroWarning,
    *,
    stacklevel: int = 2,
) -> None:
    """Emit a Tenro warning pointing to the caller's location.

    Args:
        message: Warning message to display.
        category: Warning category class. Defaults to TenroWarning.
        stacklevel: Stack frames to skip. Default (2) points to the caller
            of the function that calls warn(). Increase for nested helpers.
    """
    _warnings.warn(message, category, stacklevel=stacklevel)
