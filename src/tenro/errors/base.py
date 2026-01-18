# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Base exceptions for Tenro SDK.

Error Hierarchy:
- TenroError (Exception) - Base for all SDK errors
- TenroValidationError (TenroError) - Bad API usage/parameters
- TenroConfigError (TenroError) - Configuration/setup mistakes
- TenroConstructError (TenroError) - Test harness errors
- TenroAgentError (TenroError) - Agent execution errors
- TenroProviderConfigError (TenroConfigError) - Provider setup issues
- TenroProviderRuntimeError (TenroError) - Provider runtime issues
- TenroVerificationError (AssertionError) - Test verification failures

TenroVerificationError inherits AssertionError so pytest shows FAIL (not ERROR).
"""

from __future__ import annotations


class TenroError(Exception):
    """Base exception for all Tenro errors."""


class TenroValidationError(TenroError):
    """Raised when API usage or parameters are invalid."""


class TenroConfigError(TenroError):
    """Raised when configuration or setup is invalid."""


class TenroConstructError(TenroError):
    """Base exception for Construct test harness errors."""


class TenroAgentError(TenroError):
    """Base exception for agent-related errors."""


class TenroAgentRecursionError(TenroAgentError):
    """Raised when agent exceeds maximum nesting depth.

    This usually indicates an infinite loop between agents
    (e.g., Agent A calls Agent B, which calls Agent A again).
    """


class TenroProviderConfigError(TenroConfigError):
    """Raised when provider configuration is invalid."""


class TenroProviderRuntimeError(TenroError):
    """Raised when a provider encounters a runtime error."""


class TenroVerificationError(AssertionError):
    """Raised when test verification fails.

    Inherits from AssertionError so pytest shows FAIL (not ERROR),
    matching expected test semantics.
    """


class TenroMissingLLMCallError(TenroVerificationError):
    """Raised when a linked LLM function doesn't call the provider.

    The decorated function executed but no HTTP request was made to the
    LLM provider, so the configured simulation was never used.
    """


class TenroUnexpectedLLMCallError(TenroVerificationError):
    """Raised when an unmatched request hits a blocked LLM domain.

    This error protects against accidentally calling real LLM APIs during
    tests. If a request to a known LLM provider (e.g., api.openai.com) is
    made without a matching simulation, this error is raised immediately.

    Attributes:
        domain: The blocked domain that was accessed (e.g., "api.openai.com").
        url: The full URL of the blocked request.
    """

    def __init__(self, domain: str, url: str) -> None:
        """Initialize with domain and URL information.

        Args:
            domain: The blocked domain (e.g., "api.openai.com").
            url: The full URL of the blocked request.
        """
        self.domain = domain
        self.url = url
        super().__init__(
            f"Unmatched request to '{domain}' would hit real API: {url}\n\n"
            "To fix:\n"
            "  1. Add simulation: construct.simulate_llm(provider=..., response=...)\n"
            "  2. Or allow real calls: Construct(allow_real_llm_calls=True)"
        )


class TenroUnusedSimulationError(TenroVerificationError):
    """Raised when a simulation was registered but never triggered.

    The `simulate_llm()` call set up a simulation, but the code path that would
    trigger it was never executed.
    """


class TenroSimulationCoverageError(TenroVerificationError):
    """Raised when simulation coverage requirements are not met.

    Coverage errors indicate incomplete test execution where expected
    simulations were not triggered.
    """


class TenroSimulationUsageError(TenroConfigError):
    """Raised when simulation API is used incorrectly.

    This error is raised when simulate() is called without an active Construct
    context. All simulation calls must be made within a Construct context.
    """


class TenroLateImportWarning(UserWarning):
    """Warning issued when modules are imported before Tenro patching.

    This warning indicates best-effort patching was applied but stale
    references may exist. The warning is informational; use strict mode
    (--tenro-strict-patch) to convert this to an error.
    """


class TenroPatchingSetupError(TenroConfigError):
    """Raised when module patching fails at pytest startup.

    This error indicates that modules were imported before Tenro could
    install its import hooks. Stale references may exist that bypass
    simulation.

    Attributes:
        late_modules: List of module names that were already imported.
    """

    def __init__(self, late_modules: list[str], message: str | None = None) -> None:
        """Initialize with late module information.

        Args:
            late_modules: Modules already imported before Tenro installed.
            message: Optional custom error message.
        """
        self.late_modules = late_modules
        if message is None:
            message = (
                f"Modules {late_modules} already imported before Tenro. "
                "Stale references may exist. "
                "Use 'python -m tenro.pytest' or ensure Tenro loads first."
            )
        super().__init__(message)
