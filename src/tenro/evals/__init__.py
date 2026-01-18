# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Fast, deterministic evaluation utilities for AI agent testing.

This module provides a library of transparent evaluation functions that can be
imported and used explicitly in your tests. All evaluators are pure functions
with no side effects, giving you complete control over when and how to run them.

Examples:
    >>> import tenro.evals
    >>> result = tenro.evals.exact_match("Hello", "hello", case_sensitive=False)
    >>> assert result.passed
"""

from __future__ import annotations

from tenro.evals.types import EvalResult
from tenro.evals.validators import exact_match, json_valid, regex_match

__all__ = ["EvalResult", "exact_match", "json_valid", "regex_match"]
