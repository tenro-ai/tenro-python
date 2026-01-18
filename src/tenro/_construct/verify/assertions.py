# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Type-safe assertion models for testing."""

from __future__ import annotations

from pydantic import model_validator

from tenro._core.model_base import BaseModel


class CallCountAssertion(BaseModel):
    """Type-safe call count assertion configuration.

    Validates count parameters and provides matching logic.
    Supports exact counts, ranges, and open-ended constraints.

    Attributes:
        count: Exact expected call count.
        min: Minimum expected call count.
        max: Maximum expected call count.

    Examples:
        >>> # Exact count
        >>> assertion = CallCountAssertion(count=2)
        >>> assertion.matches(2)  # True
        >>> assertion.matches(3)  # False

        >>> # Range
        >>> assertion = CallCountAssertion(min=2, max=4)
        >>> assertion.matches(3)  # True
        >>> assertion.matches(5)  # False

        >>> # At least
        >>> assertion = CallCountAssertion(min=2)
        >>> assertion.matches(3)  # True

        >>> # At most
        >>> assertion = CallCountAssertion(max=3)
        >>> assertion.matches(2)  # True
    """

    count: int | None = None
    min: int | None = None
    max: int | None = None

    @model_validator(mode="after")
    def validate_count_params(self) -> CallCountAssertion:
        """Ensure count is mutually exclusive with min/max.

        Returns:
            Validated CallCountAssertion instance.

        Raises:
            ValueError: If both count and min/max are specified.
            ValueError: If min > max.
        """
        has_exact = self.count is not None
        has_range = self.min is not None or self.max is not None

        if has_exact and has_range:
            msg = (
                "Cannot specify both 'count' and 'min'/'max'. "
                "Use 'count' for exact assertions or 'min'/'max' for ranges."
            )
            raise ValueError(msg)

        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min ({self.min}) cannot be greater than max ({self.max})")

        return self

    def matches(self, actual: int) -> bool:
        """Check if actual count matches assertion criteria.

        Args:
            actual: The actual count to check.

        Returns:
            `True` if actual count satisfies the criteria.
        """
        if self.count is not None:
            return actual == self.count

        if self.min is not None and actual < self.min:
            return False

        if self.max is not None and actual > self.max:
            return False

        # If no constraints specified, assert at least one
        if self.count is None and self.min is None and self.max is None:
            return actual > 0

        return True

    def error_message(self, actual: int, context: str = "") -> str:
        """Generate descriptive error message.

        Args:
            actual: The actual count found.
            context: Optional context string (e.g., " with provider=openai").

        Returns:
            Formatted error message with expected vs actual.
        """
        expected = self._format_expected()
        return f"Expected {expected} call(s){context}, but found {actual}."

    def _format_expected(self) -> str:
        """Format expected count as human-readable string."""
        if self.count is not None:
            return f"exactly {self.count}"
        if self.min is not None and self.max is not None:
            return f"between {self.min} and {self.max}"
        if self.min is not None:
            return f"at least {self.min}"
        if self.max is not None:
            return f"at most {self.max}"
        return "at least 1"


__all__ = ["CallCountAssertion"]
