# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Evaluation types for Tenro SDK."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from tenro._core.model_base import BaseModel

EvalScore = float


class EvalResult(BaseModel):
    """Result from an evaluation.

    This model is frozen (immutable) to ensure evaluation data
    cannot be modified after creation.

    The instance is truthy if `passed=True`, allowing natural pytest assertions:
        >>> data = exact_match("hello", "hello")
        >>> assert data  # Equivalent to: assert data.passed

    Attributes:
        score: Numeric evaluation score.
        passed: Whether the evaluation passed.
        details: Optional metadata with evaluator-specific details.
    """

    model_config = ConfigDict(frozen=True)

    score: float
    passed: bool
    details: dict[str, Any] | None = Field(default=None)

    def __bool__(self) -> bool:
        """Allow result to be used in boolean context.

        Returns:
            `True` if evaluation passed, `False` otherwise.

        Examples:
            >>> result = exact_match("hello", "hello")
            >>> assert result  # Works! Checks result.passed
        """
        return self.passed


__all__ = ["EvalResult", "EvalScore"]
