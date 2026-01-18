# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Simulation-specific exceptions for Tenro SDK.

These exceptions are raised during simulation setup and execution to provide
clear diagnostics when simulation configuration is invalid or fails at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from tenro.errors.base import TenroConfigError, TenroError


@dataclass
class SimulationDiagnostic:
    """Structured diagnostic information for simulation errors.

    Provides machine-readable fields for tooling and human-readable messages.
    """

    target_path: str
    """Canonical path of the simulation target (e.g., 'mymodule.my_tool')."""

    target_type: str
    """Type of the target (e.g., 'function', 'method', 'class', 'builtin')."""

    is_linked: bool
    """Whether the target is a linked callable (@link_tool, @link_agent, @link_llm)."""

    failure_reason: str
    """Technical explanation of why the operation failed."""

    recommended_fix: str
    """Actionable guidance for resolving the error."""


class TenroSimulationSetupError(TenroConfigError):
    """Raised when simulation setup fails due to invalid target.

    This error occurs during `simulate()` calls when the target cannot be
    simulated. Common causes:

    - Target is not decorated with @link_tool, @link_agent, or @link_llm
    - Target is a builtin or C-extension that cannot be intercepted
    - Target identity cannot be resolved

    Attributes:
        diagnostic: Structured diagnostic with target info and fix suggestions.
    """

    def __init__(self, message: str, diagnostic: SimulationDiagnostic) -> None:
        """Initialize with message and diagnostic info.

        Args:
            message: Human-readable error message.
            diagnostic: Structured diagnostic information.
        """
        self.diagnostic = diagnostic
        super().__init__(message)

    @classmethod
    def not_linked(
        cls, target_path: str, target_type: str = "function"
    ) -> TenroSimulationSetupError:
        """Create error for non-linked target.

        Args:
            target_path: The target's dotted path or repr.
            target_type: Type of the target (function, method, class, etc.).

        Returns:
            Configured error with diagnostic.
        """
        diagnostic = SimulationDiagnostic(
            target_path=target_path,
            target_type=target_type,
            is_linked=False,
            failure_reason="Target is not linked; capture-safe interception unavailable.",
            recommended_fix="Decorate with @link_tool, @link_agent, or @link_llm.",
        )
        return cls(
            f"Cannot simulate '{target_path}': not a linked callable.\n\n"
            f"Reason: {diagnostic.failure_reason}\n"
            f"Fix: {diagnostic.recommended_fix}",
            diagnostic=diagnostic,
        )

    @classmethod
    def not_patchable(
        cls, target_path: str, reason: str, target_type: str = "function"
    ) -> TenroSimulationSetupError:
        """Create error for non-patchable target (registered function simulation).

        Args:
            target_path: The target's dotted path or repr.
            reason: Why the target is not patchable.
            target_type: Type of the target (function, method, class, etc.).

        Returns:
            Configured error with diagnostic.
        """
        diagnostic = SimulationDiagnostic(
            target_path=target_path,
            target_type=target_type,
            is_linked=False,
            failure_reason=f"Target is not patchable: {reason}",
            recommended_fix=(
                "Use @link_tool decorator, HTTP boundary interception, "
                "or wrap the function in a patchable Python function."
            ),
        )
        return cls(
            f"Cannot register '{target_path}' for simulation: not patchable.\n\n"
            f"Reason: {reason}\n"
            f"Fix: {diagnostic.recommended_fix}",
            diagnostic=diagnostic,
        )


class TenroSimulationExecutionError(TenroError):
    """Raised when simulation execution produces unexpected callable kind.

    This error occurs at call time when a simulation rule's side_effect or
    configured response produces the wrong type for the wrapper's expected kind:

    - Sync wrapper expects non-awaitable, non-generator value
    - Async wrapper expects awaitable
    - Generator wrapper expects Generator
    - Async generator wrapper expects AsyncGenerator

    Attributes:
        expected_kind: The callable kind the wrapper expected.
        actual_kind: The callable kind that was produced.
        target_path: The simulation target's canonical path.
    """

    def __init__(
        self,
        target_path: str,
        expected_kind: str,
        actual_kind: str,
    ) -> None:
        """Initialize with kind mismatch information.

        Args:
            target_path: The simulation target's canonical path.
            expected_kind: Expected kind (sync/async/gen/asyncgen).
            actual_kind: Actual kind that was produced.
        """
        self.target_path = target_path
        self.expected_kind = expected_kind
        self.actual_kind = actual_kind
        super().__init__(
            f"Simulation for '{target_path}' returned wrong callable kind.\n\n"
            f"Expected: {expected_kind}\n"
            f"Got: {actual_kind}\n\n"
            "Ensure side_effect/result matches the target's signature.",
        )
