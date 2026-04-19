# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Tests for the @tenro.simulate decorator."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path


class TestSimulateDecorator:
    """Verify @tenro.simulate works as both module and decorator."""

    def _run_pytest(self, tmp_path: Path, test_code: str) -> subprocess.CompletedProcess[str]:
        """Write a test file and run pytest on it in a subprocess.

        Args:
            tmp_path: Pytest temporary directory.
            test_code: Python test source to write and execute.

        Returns:
            Completed subprocess result.
        """
        test_file = tmp_path / "test_check.py"
        test_file.write_text(test_code)
        return subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )

    def test_bare_decorator_on_function(self, tmp_path: Path) -> None:
        """@tenro.simulate activates the construct fixture on a function."""
        result = self._run_pytest(
            tmp_path,
            """
import tenro

@tenro.simulate
def test_has_construct(construct):
    assert construct is not None
""",
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "1 passed" in result.stdout

    def test_bare_decorator_on_class(self, tmp_path: Path) -> None:
        """@tenro.simulate activates the construct fixture on a class."""
        result = self._run_pytest(
            tmp_path,
            """
import tenro

@tenro.simulate
class TestMyAgent:
    def test_has_construct(self, construct):
        assert construct is not None
""",
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "1 passed" in result.stdout

    def test_empty_parens_decorator(self, tmp_path: Path) -> None:
        """@tenro.simulate() works with empty parentheses."""
        result = self._run_pytest(
            tmp_path,
            """
import tenro

@tenro.simulate()
def test_has_construct(construct):
    assert construct is not None
""",
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "1 passed" in result.stdout

    def test_parametrised_decorator(self, tmp_path: Path) -> None:
        """@tenro.simulate(allow_real_llm_calls=True) configures the Construct."""
        result = self._run_pytest(
            tmp_path,
            """
import tenro

@tenro.simulate(allow_real_llm_calls=True)
def test_with_option(construct):
    assert construct._http_interceptor._blocked_domains == []
""",
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "1 passed" in result.stdout

    def test_stacked_markers_merge(self, tmp_path: Path) -> None:
        """Class + method markers merge; nearest scope wins per key."""
        result = self._run_pytest(
            tmp_path,
            """
import tenro

@tenro.simulate(allow_real_llm_calls=True)
class TestStacked:
    @tenro.simulate(fail_unused=True)
    def test_both_options(self, construct):
        # Method-level fail_unused
        assert construct._fail_unused is True
        # Class-level allow_real_llm_calls inherited
        assert construct._http_interceptor._blocked_domains == []
""",
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "1 passed" in result.stdout

    def test_invalid_kwarg_type_rejected(self, tmp_path: Path) -> None:
        """String where bool expected raises ValueError."""
        result = self._run_pytest(
            tmp_path,
            """
import tenro

@tenro.simulate(allow_real_llm_calls="true")
def test_bad_type(construct):
    pass
""",
        )
        assert result.returncode != 0
        assert "expected bool" in result.stdout

    def test_unknown_kwarg_rejected(self, tmp_path: Path) -> None:
        """Unknown kwargs raise ValueError."""
        result = self._run_pytest(
            tmp_path,
            """
import tenro

@tenro.simulate(bogus_option=True)
def test_unknown(construct):
    pass
""",
        )
        assert result.returncode != 0
        assert "unexpected keyword argument" in result.stdout

    def test_submodule_imports_still_work(self) -> None:
        """from tenro.simulate import llm/tool/agent is unaffected."""
        from tenro.simulate import agent, llm, register, tool

        assert hasattr(llm, "simulate")
        assert hasattr(tool, "simulate")
        assert hasattr(agent, "simulate")
        assert callable(register)

    def test_module_is_callable(self) -> None:
        """tenro.simulate is callable (the decorator)."""
        import tenro.simulate

        assert callable(tenro.simulate)

    def test_module_is_still_a_module(self) -> None:
        """tenro.simulate is still a module for inspect/importlib."""
        import tenro.simulate

        assert inspect.ismodule(tenro.simulate)

    def test_accessible_via_tenro_package(self) -> None:
        """import tenro; tenro.simulate works via lazy loading."""
        import tenro

        sim = tenro.simulate
        assert callable(sim)
        assert inspect.ismodule(sim)
