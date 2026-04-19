# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pytest plugin entry points for Tenro.

Auto-loads with pytest via pytest11 entrypoint. The plugin:
1. Installs PatchEngine early (pytest_load_initial_conftests) for module patching
2. Provides fixtures for test harness (construct, make_construct)
3. Supports tracing visualization (--tenro-print-trace)

The PatchEngine patches already-imported target modules in sys.modules
at startup. Real simulation uses transport-level seams (respx) and
``@link_*`` decorators, not import-time patching.

Disable autopatching via:
- --tenro-no-autopatch flag
- TENRO_DISABLE_PYTEST_AUTOPATCH=1 environment variable
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

    from _pytest.config import Config
    from _pytest.config.argparsing import Parser

    from tenro.trace.config import TraceConfig

# Import fixtures so pytest can discover them
from tenro.pytest_plugin.fixtures import construct

# Module-level state
_plugin_enabled = False
_trace_enabled = False
_trace_config: TraceConfig | None = None
_autopatch_enabled = True
_patch_engine_installed = False
_fail_unused = False

__all__ = ["construct", "is_fail_unused"]


def is_fail_unused() -> bool:
    """Check if fail-unused mode is enabled.

    Returns:
        True if --tenro-fail-unused flag is set or
        TENRO_FAIL_UNUSED env var is truthy.
    """
    return _fail_unused


_CLI_OPTIONS: tuple[tuple[str, str], ...] = (
    ("--tenro", "Enable Tenro tracing and constructing for agent tests"),
    ("--tenro-print-trace", "Print trace visualization after each test"),
    ("--tenro-trace", "Deprecated alias for --tenro-print-trace"),
    ("--tenro-no-autopatch", "Disable automatic module patching at startup"),
    ("--tenro-strict-patch", "Raise error if modules are already imported (strict mode)"),
    ("--tenro-fail-unused", "Raise error (instead of warning) for unused simulations"),
)


def pytest_addoption(parser: Parser) -> None:
    """Add Tenro CLI options to pytest.

    Args:
        parser: Pytest argument parser.
    """
    group = parser.getgroup("tenro")
    for flag, help_text in _CLI_OPTIONS:
        group.addoption(flag, action="store_true", default=False, help=help_text)


def pytest_load_initial_conftests(
    early_config: Config,
    parser: Parser,
    args: list[str],
) -> None:
    """Earliest hook - install PatchEngine BEFORE test files are imported.

    This runs before pytest collects test files, ensuring import hooks
    are in place before any test code captures references to third-party modules.

    Args:
        early_config: Early pytest configuration.
        parser: Argument parser.
        args: Command line arguments.
    """
    global _autopatch_enabled, _patch_engine_installed
    del early_config, parser

    truthy_values = ("true", "1", "yes")
    env_disabled = os.getenv("TENRO_DISABLE_PYTEST_AUTOPATCH", "").lower() in truthy_values
    cli_disabled = "--tenro-no-autopatch" in args
    if env_disabled or cli_disabled:
        _autopatch_enabled = False
        return

    strict = "--tenro-strict-patch" in args
    if _install_patch_engine(strict=strict):
        _patch_engine_installed = True


def _install_patch_engine(*, strict: bool) -> bool:
    """Install the Tenro PatchEngine for import-time instrumentation.

    Args:
        strict: If True, re-raise install failures instead of warning.

    Returns:
        True if the engine was installed, False on ImportError.
    """
    try:
        from tenro._patching.engine import get_engine
        from tenro._patching.patchers import register_all_patchers

        register_all_patchers()
        get_engine().install(strict=strict)
        return True
    except ImportError:
        return False
    except Exception as e:
        if strict:
            raise
        _warn_patch_install_failed(e)
        return False


def _warn_patch_install_failed(error: Exception) -> None:
    """Emit a non-strict warning when PatchEngine installation fails.

    Fail loudly (warn, do not silently ignore) so users notice that
    captured references may bypass simulation.
    """
    import warnings

    from tenro.errors import TenroPatchingWarning

    warnings.warn(
        f"Tenro PatchEngine installation failed: {error}. "
        "Simulation may not work for captured references. "
        "Use --tenro-strict-patch to make this an error.",
        TenroPatchingWarning,
        stacklevel=2,
    )


def pytest_configure(config: Config) -> None:
    """Configure Tenro plugin when pytest starts.

    Registers markers and toggles plugin state based on --tenro or
    TENRO_ENABLED. Trace output is enabled via --tenro-print-trace or TENRO_PRINT_TRACE.

    Args:
        config: Pytest configuration object.
    """
    global _plugin_enabled, _trace_enabled, _trace_config, _fail_unused

    # Register custom marker
    config.addinivalue_line(
        "markers",
        "tenro: Mark test for Tenro tracing and constructing",
    )

    truthy_values = ("true", "1", "yes")

    _plugin_enabled = (
        config.getoption("--tenro", default=False)
        or os.getenv("TENRO_ENABLED", "").lower() in truthy_values
    )

    _trace_enabled = (
        config.getoption("--tenro-print-trace", default=False)
        or config.getoption("--tenro-trace", default=False)
        or os.getenv("TENRO_PRINT_TRACE", "").lower() in truthy_values
        or os.getenv("TENRO_TRACE", "").lower() in truthy_values
    )

    if _trace_enabled:
        from tenro.trace.config import get_trace_config

        _trace_config = get_trace_config()

    _fail_unused = (
        config.getoption("--tenro-fail-unused", default=False)
        or os.getenv("TENRO_FAIL_UNUSED", "").lower() in truthy_values
    )


# Re-export the makereport hook from block_report so pytest discovers it via
# this module (the entry point loads ``tenro.pytest_plugin.plugin``).
from tenro.pytest_plugin.block_report import pytest_runtest_makereport  # noqa: E402,F401


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item: pytest.Item) -> Generator[None, None, None]:
    """Teardown hook called after each test.

    Renders trace visualization if --tenro-print-trace is enabled and the test
    used the construct fixture. Implemented as a hookwrapper so the
    renderer runs *after* all fixture finalizers — spans emitted during
    fixture teardown (e.g. ``with construct.link_agent(...)`` after
    ``yield`` in a user fixture) are captured in the trace output.

    A strong reference to the construct is pinned before ``yield`` so
    the renderer can still access it after the ``construct`` fixture's
    finalizer deletes ``item._tenro_construct``.

    Args:
        item: Test item that just ran.

    Yields:
        Control to downstream teardown hooks and fixture finalizers.
    """
    captured = getattr(item, "_tenro_construct", None) if _trace_enabled else None

    yield

    if captured is None:
        return
    agents = captured.agent_runs
    if not agents:
        return

    from tenro.trace import TraceRenderer

    renderer = TraceRenderer(config=_trace_config)
    renderer.render(agents, test_name=item.name)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Called at session start before collection.

    Clears construct registry to ensure fresh state for this test run.

    Args:
        session: Pytest session object (required by hook signature).
    """
    del session
    from tenro.pytest_plugin.construct_registry import clear_constructs

    clear_constructs()


def pytest_unconfigure(config: Config) -> None:
    """Called when pytest exits.

    Resets the PatchEngine install flag. Already-patched modules remain
    instrumented until process exit.

    Args:
        config: Pytest configuration object (required by hook signature).
    """
    del config
    global _patch_engine_installed

    if _patch_engine_installed:
        try:
            from tenro._patching.engine import get_engine

            engine = get_engine()
            engine.uninstall()
            _patch_engine_installed = False
        except ImportError:
            pass

    # Clear OTel config so it doesn't leak between test runs
    from tenro._init import reset

    reset()
