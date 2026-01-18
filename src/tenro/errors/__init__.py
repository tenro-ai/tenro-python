# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Exceptions and warnings for Tenro SDK."""

from __future__ import annotations

from tenro.errors.base import (
    TenroAgentError,
    TenroAgentRecursionError,
    TenroConfigError,
    TenroConstructError,
    TenroError,
    TenroMissingLLMCallError,
    TenroProviderConfigError,
    TenroProviderRuntimeError,
    TenroSimulationCoverageError,
    TenroSimulationUsageError,
    TenroUnexpectedLLMCallError,
    TenroUnusedSimulationError,
    TenroValidationError,
    TenroVerificationError,
)
from tenro.errors.simulation import (
    SimulationDiagnostic,
    TenroSimulationExecutionError,
    TenroSimulationSetupError,
)
from tenro.errors.warnings import (
    TenroCoercionWarning,
    TenroConfigWarning,
    TenroDeprecationWarning,
    TenroFutureWarning,
    TenroLateImportWarning,
    TenroPatchingWarning,
    TenroPluginWarning,
    TenroTracingWarning,
    TenroUnusedSimulationWarning,
    TenroWarning,
    warn,
)

__all__ = [
    "SimulationDiagnostic",
    "TenroAgentError",
    "TenroAgentRecursionError",
    "TenroCoercionWarning",
    "TenroConfigError",
    "TenroConfigWarning",
    "TenroConstructError",
    "TenroDeprecationWarning",
    "TenroError",
    "TenroFutureWarning",
    "TenroLateImportWarning",
    "TenroMissingLLMCallError",
    "TenroPatchingWarning",
    "TenroPluginWarning",
    "TenroProviderConfigError",
    "TenroProviderRuntimeError",
    "TenroSimulationCoverageError",
    "TenroSimulationExecutionError",
    "TenroSimulationSetupError",
    "TenroSimulationUsageError",
    "TenroTracingWarning",
    "TenroUnexpectedLLMCallError",
    "TenroUnusedSimulationError",
    "TenroUnusedSimulationWarning",
    "TenroValidationError",
    "TenroVerificationError",
    "TenroWarning",
    "warn",
]
