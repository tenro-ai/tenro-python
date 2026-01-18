# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Fast, deterministic evaluation utilities for AI agent testing.

This module provides a library of transparent evaluation functions that can be
imported and used explicitly in your tests. All evaluators are pure functions
with no side effects, giving you complete control over when and how to run them.

Common Use Cases:
    - Validating agent outputs in pytest tests
    - Running local evaluations before cloud sync
    - Creating custom test assertions with detailed feedback
    - Extracting structured data from conversational LLM outputs

Examples:
    >>> import tenro.evals
    >>>
    >>> # Strict validation
    >>> result = tenro.evals.exact_match(
    ...     output="Hello world",
    ...     expected="Hello world",
    ...     case_sensitive=False
    ... )
    >>> assert result.passed
"""

from __future__ import annotations

import json
import re

from tenro.evals.types import EvalResult


def exact_match(
    output: str,
    expected: str,
    case_sensitive: bool = True,
) -> EvalResult:
    """Validate that agent output exactly matches the expected value.

    Performs string comparison after stripping leading/trailing whitespace.
    Useful for validating deterministic outputs like classification labels,
    structured responses, or templated text.

    Args:
        output: The agent's output string to evaluate.
        expected: The expected output string.
        case_sensitive: Whether to perform case-sensitive comparison.
            Defaults to `True`.

    Returns:
        EvalResult with score 1.0 if strings match, 0.0 otherwise. Details
        contains the original output, expected value, and comparison mode.

    Examples:
        >>> # Case-insensitive comparison
        >>> result = exact_match("Hello World", "hello world", case_sensitive=False)
        >>> assert result.passed
        >>> assert result.score == 1.0
        >>>
        >>> # Case-sensitive comparison (default)
        >>> result = exact_match("Hello World", "hello world")
        >>> assert not result.passed
    """
    output_compare = output.strip() if case_sensitive else output.strip().lower()
    expected_compare = expected.strip() if case_sensitive else expected.strip().lower()

    passed = output_compare == expected_compare
    score = 1.0 if passed else 0.0

    return EvalResult(
        score=score,
        passed=passed,
        details={
            "output": output,
            "expected": expected,
            "case_sensitive": case_sensitive,
        },
    )


def regex_match(output: str, pattern: str) -> EvalResult:
    r"""Validate that agent output matches a regular expression pattern.

    Uses Python's re.search() to find the pattern anywhere in the output.
    Useful for validating outputs with variable content like phone numbers,
    emails, dates, or structured text with dynamic fields.

    Args:
        output: The agent's output string to evaluate.
        pattern: Regular expression pattern to search for (Python regex syntax).

    Returns:
        EvalResult with score 1.0 if pattern is found, 0.0 otherwise.
        The details field contains the output, pattern, and match status.

    Examples:
        >>> # Validate phone number format
        >>> result = regex_match("My number is 555-1234", r"\\d{3}-\\d{4}")
        >>> assert result.passed
        >>>
        >>> # Validate email presence
        >>> result = regex_match(
        ...     "Contact: user@example.com",
        ...     r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
        ... )
        >>> assert result.passed
    """
    match = re.search(pattern, output)
    passed = match is not None
    score = 1.0 if passed else 0.0

    return EvalResult(
        score=score,
        passed=passed,
        details={
            "output": output,
            "pattern": pattern,
            "matched": passed,
        },
    )


def json_valid(output: str) -> EvalResult:
    """Validate that output is strict, pure JSON with no additional text.

    This is a strict validator that fails if there is any "chatty" text
    around the JSON (e.g., "Sure! Here you go: {...}").

    Args:
        output: Agent output to evaluate.

    Returns:
        EvalResult with score 1.0 if valid JSON, 0.0 otherwise.
        The parsed JSON object is included in details["parsed"] on success.

    Examples:
        >>> result = json_valid('{"key": "value"}')
        >>> assert result.passed
        >>> assert result.details["parsed"] == {"key": "value"}
        >>>
        >>> result = json_valid("Sure! Here is your JSON: {...}")
        >>> assert not result.passed  # Fails due to chatty text
    """
    try:
        parsed = json.loads(output)
        return EvalResult(
            score=1.0,
            passed=True,
            details={"valid": True, "parsed": parsed},
        )
    except json.JSONDecodeError as e:
        return EvalResult(
            score=0.0,
            passed=False,
            details={"valid": False, "error": str(e)},
        )


__all__ = ["exact_match", "json_valid", "regex_match"]
