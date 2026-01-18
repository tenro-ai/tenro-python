# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Tests for Tenro pytest plugin CLI options."""

from __future__ import annotations

import subprocess
import sys

import pytest


class TestPytestPluginOptions:
    """Test pytest CLI options for Tenro plugin."""

    def test_no_autopatch_flag_disables_patching(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify --tenro-no-autopatch disables automatic module patching."""
        # Create a test file that checks if patching is disabled
        test_file = tmp_path / "test_check.py"
        test_file.write_text(
            """
import tenro.pytest_plugin.plugin as plugin

def test_autopatch_disabled():
    assert plugin._autopatch_enabled is False
"""
        )

        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "--tenro-no-autopatch", "-v"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert result.returncode == 0, f"Test failed: {result.stdout}\n{result.stderr}"
        assert "1 passed" in result.stdout

    def test_autopatch_enabled_by_default(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify autopatch is enabled by default."""
        test_file = tmp_path / "test_check.py"
        test_file.write_text(
            """
import tenro.pytest_plugin.plugin as plugin

def test_autopatch_enabled():
    assert plugin._autopatch_enabled is True
"""
        )

        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert result.returncode == 0, f"Test failed: {result.stdout}\n{result.stderr}"
        assert "1 passed" in result.stdout

    def test_strict_expectations_flag_recognized(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify --tenro-strict-expectations flag is recognized."""
        test_file = tmp_path / "test_check.py"
        test_file.write_text(
            """
def test_placeholder():
    pass
"""
        )

        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "--tenro-strict-expectations", "-v"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        # Should not error with "unrecognized arguments"
        assert "unrecognized arguments" not in result.stderr
        assert result.returncode == 0, f"Test failed: {result.stdout}\n{result.stderr}"

    def test_trace_flag_recognized(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify --tenro-trace flag is recognized."""
        test_file = tmp_path / "test_check.py"
        test_file.write_text(
            """
def test_placeholder():
    pass
"""
        )

        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "--tenro-trace", "-v"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        # Should not error with "unrecognized arguments"
        assert "unrecognized arguments" not in result.stderr
        assert result.returncode == 0, f"Test failed: {result.stdout}\n{result.stderr}"

    def test_env_var_disables_autopatch(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify TENRO_DISABLE_PYTEST_AUTOPATCH=1 disables patching."""
        test_file = tmp_path / "test_check.py"
        test_file.write_text(
            """
import tenro.pytest_plugin.plugin as plugin

def test_autopatch_disabled_via_env():
    assert plugin._autopatch_enabled is False
"""
        )

        import os

        env = os.environ.copy()
        env["TENRO_DISABLE_PYTEST_AUTOPATCH"] = "1"

        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=env,
        )
        assert result.returncode == 0, f"Test failed: {result.stdout}\n{result.stderr}"
        assert "1 passed" in result.stdout
