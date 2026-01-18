# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Custom pytest markers for Tenro.

This module defines custom markers for controlling Tenro behavior
on a per-test basis.
"""

from __future__ import annotations

import pytest

tenro = pytest.mark.tenro
"""Mark a test for Tenro tracing and constructing.

Examples:
    >>> @pytest.mark.tenro
    ... def test_my_agent():
    ...     # This test will be traced by Tenro
    ...     ...
"""

__all__ = ["tenro"]
