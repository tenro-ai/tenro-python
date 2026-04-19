# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pytest integration utilities (deprecated).

.. deprecated:: 0.3.0
    Use ``@tenro.simulate`` instead. Will be removed in 0.5.0.

    Before::

        from tenro.testing import tenro

        @tenro
        def test_my_agent(): ...

    After::

        import tenro

        @tenro.simulate
        def test_my_agent(): ...
"""

from __future__ import annotations

import warnings


def __getattr__(name: str) -> object:
    """Emit deprecation warning on access to ``tenro.testing.tenro``."""
    if name == "tenro":
        warnings.warn(
            "tenro.testing.tenro is deprecated. "
            "Use 'import tenro' and '@tenro.simulate' instead. "
            "Will be removed in tenro 0.5.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        import pytest

        return pytest.mark.usefixtures("construct")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["tenro"]  # noqa: F822 — dynamic via __getattr__
