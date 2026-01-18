# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pytest plugin entry points for Tenro.

Auto-loads with pytest via pytest11 entrypoint. The plugin:
1. Installs PatchEngine early (pytest_load_initial_conftests) for module patching
2. Provides fixtures for test harness (construct, make_construct)
3. Supports tracing visualization (--tenro-trace)

The PatchEngine uses double-tap approach:
- Immediate pass: patch already-imported modules in sys.modules
- Forward pass: install sys.meta_path hook for future imports

Disable autopatching via:
- --tenro-no-autopatch flag
- TENRO_DISABLE_PYTEST_AUTOPATCH=1 environment variable
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser

# Import fixtures so pytest can discover them
from tenro.pytest_plugin.fixtures import construct

# Module-level state
_plugin_enabled = False
_trace_enabled = False
_autopatch_enabled = True
_patch_engine_installed = False
_strict_expectations = False

__all__ = ["construct", "is_strict_expectations"]


def is_strict_expectations() -> bool:
    """Check if strict expectations mode is enabled.

    Returns:
        True if --tenro-strict-expectations flag is set or
        TENRO_STRICT_EXPECTATIONS env var is truthy.
    """
    return _strict_expectations


def pytest_addoption(parser: Parser) -> None:
    """Add Tenro CLI options to pytest.

    Args:
        parser: Pytest argument parser.
    """
    group = parser.getgroup("tenro")
    group.addoption(
        "--tenro",
        action="store_true",
        default=False,
        help="Enable Tenro tracing and constructing for agent tests",
    )
    group.addoption(
        "--tenro-trace",
        action="store_true",
        default=False,
        help="Print trace visualization after each test",
    )
    group.addoption(
        "--tenro-no-autopatch",
        action="store_true",
        default=False,
        help="Disable automatic module patching at startup",
    )
    group.addoption(
        "--tenro-strict-patch",
        action="store_true",
        default=False,
        help="Raise error if modules are already imported (strict mode)",
    )
    group.addoption(
        "--tenro-strict-expectations",
        action="store_true",
        default=False,
        help="Raise error (instead of warning) for unused simulations",
    )


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

    # Suppress unused argument warnings
    del early_config, parser

    # Allow users to disable autopatch via env var or CLI flag (for opt-out cases)
    truthy_values = ("true", "1", "yes")
    env_disabled = os.getenv("TENRO_DISABLE_PYTEST_AUTOPATCH", "").lower() in truthy_values
    cli_disabled = "--tenro-no-autopatch" in args

    if env_disabled or cli_disabled:
        _autopatch_enabled = False
        return

    # Support explicit strict mode for tighter verification
    strict = "--tenro-strict-patch" in args

    # Install PatchEngine
    try:
        from tenro._patching.engine import get_engine
        from tenro._patching.patchers import register_all_patchers

        register_all_patchers()
        engine = get_engine()
        engine.install(strict=strict)
        _patch_engine_installed = True
    except ImportError:
        # Patching module not available - skip silently
        pass
    except Exception as e:
        # Fail loudly: always report patching failures
        # In strict mode: raise the exception
        # In non-strict mode: warn but continue (don't crash pytest)
        if strict:
            raise
        import warnings

        from tenro.errors import TenroPatchingWarning

        warnings.warn(
            f"Tenro PatchEngine installation failed: {e}. "
            "Simulation may not work for captured references. "
            "Use --tenro-strict-patch to make this an error.",
            TenroPatchingWarning,
            stacklevel=2,
        )


def pytest_configure(config: Config) -> None:
    """Configure Tenro plugin when pytest starts.

    Registers markers and toggles plugin state based on --tenro or
    TENRO_ENABLED. Trace output is enabled via --tenro-trace or TENRO_TRACE.

    Args:
        config: Pytest configuration object.
    """
    global _plugin_enabled, _trace_enabled, _strict_expectations

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
        config.getoption("--tenro-trace", default=False)
        or os.getenv("TENRO_TRACE", "").lower() in truthy_values
    )

    _strict_expectations = (
        config.getoption("--tenro-strict-expectations", default=False)
        or os.getenv("TENRO_STRICT_EXPECTATIONS", "").lower() in truthy_values
    )


def pytest_collection_modifyitems(config: Config, items: list[pytest.Item]) -> None:
    """Collection hook for selective tracing.

    Args:
        config: Pytest configuration.
        items: Collected test items.
    """
    if not _plugin_enabled:
        return

    pass


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Setup hook called before each test runs.

    Args:
        item: Test item about to run.
    """
    if not _plugin_enabled:
        return

    if "tenro" in item.keywords or _plugin_enabled:
        pass


def pytest_runtest_call(item: pytest.Item) -> None:
    """Hook called when test is executed.

    Args:
        item: Test item being executed.
    """
    if not _plugin_enabled:
        return

    pass


def pytest_runtest_teardown(item: pytest.Item) -> None:
    """Teardown hook called after each test.

    Renders trace visualization if --tenro-trace is enabled and the test
    used the construct fixture.

    Args:
        item: Test item that just ran.
    """
    if not _trace_enabled:
        return

    construct = getattr(item, "_tenro_construct", None)
    if construct is None:
        return

    agents = construct.agent_runs
    if not agents:
        return

    from tenro.trace import TraceConfig, TraceRenderer

    config = TraceConfig(enabled=True)
    renderer = TraceRenderer(config=config)
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

    Uninstalls the PatchEngine to restore original import behavior.

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
