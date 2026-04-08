# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Simulation validation tracking.

Validates that simulations are used during tests.
By default, warns when simulations are registered but never triggered.
Use `strict=True` to raise errors instead of warnings.
Supports `optional=True` to allow unused simulations without warnings/errors.
"""

from __future__ import annotations

import warnings

from tenro._construct.http.builders import ProviderSchemaFactory
from tenro._core.spans import LLMCall, LLMScope
from tenro.errors import TenroMissingLLMCallError, TenroUnusedSimulationError
from tenro.errors.warnings import TenroUnusedSimulationWarning


class ToolAgentTracker:
    """Tracks tool and agent simulation usage.

    Raises `TenroUnusedSimulationError` when simulations were registered but never
    called. Supports optional simulations that won't error if unused.
    """

    def __init__(self) -> None:
        self._registered: set[str] = set()  # target paths
        self._optional: set[str] = set()
        self._triggered: set[str] = set()

    def register(self, target_path: str, *, optional: bool = False) -> None:
        """Record that a simulation was registered for a target.

        Args:
            target_path: Dotted path to the target function/method.
            optional: If `True`, this simulation won't error if unused.
        """
        self._registered.add(target_path)
        if optional:
            self._optional.add(target_path)

    def mark_triggered(self, target_path: str) -> None:
        """Record that a simulation was triggered."""
        self._triggered.add(target_path)

    def validate(self, simulation_type: str, *, strict: bool = False) -> None:
        """Validate that all required simulations were used.

        Args:
            simulation_type: "tool" or "agent" for error messages.
            strict: If True, raises error. If False (default), emits warning.

        Raises:
            TenroUnusedSimulationError: If strict=True and simulation was never
                triggered.
        """
        not_triggered = self._registered - self._triggered - self._optional
        if not not_triggered:
            return

        target = sorted(not_triggered)[0]
        msg = self._build_unused_message(target, simulation_type)
        if strict:
            raise TenroUnusedSimulationError(msg)
        else:
            warnings.warn(msg, TenroUnusedSimulationWarning, stacklevel=3)

    def _build_unused_message(self, target: str, simulation_type: str) -> str:
        """Build error message for unused simulation."""
        lines = [
            f"{simulation_type.title()} simulation for '{target}' was never triggered",
            "",
            f"Your test configured simulate_{simulation_type}() for this target,",
            f"but the {simulation_type} was never called.",
            "",
            f"  Target: {target}",
            "",
            f"Fix: Remove the unused simulation or call the {simulation_type}:",
            f"  {simulation_type}.simulate({target.split('.')[-1]}, result=...)",
            f"  {target.split('.')[-1]}()  # Call the {simulation_type}",
        ]
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all tracking state."""
        self._registered.clear()
        self._optional.clear()
        self._triggered.clear()


class SimulationTracker:
    """Validates simulation registration, triggering, and execution.

    Raises `TenroMissingLLMCallError` when `@link_llm` ran but no HTTP call happened.
    Raises `TenroUnusedSimulationError` when simulations were registered but never used.

    Supports optional simulations that won't error if unused.
    """

    def __init__(self) -> None:
        self._registered: set[str] = set()
        self._optional: set[str] = set()
        self._triggered: set[str] = set()

    def register(self, provider: str, *, optional: bool = False) -> None:
        """Record that a simulation was registered for a provider.

        Args:
            provider: Provider name (e.g., "openai", "anthropic").
            optional: If `True`, this simulation won't error if unused.
        """
        self._registered.add(provider)
        if optional:
            self._optional.add(provider)

    def mark_triggered(self, provider: str) -> None:
        """Record that a simulation was triggered."""
        self._triggered.add(provider)

    def validate(
        self,
        llm_calls: list[LLMCall],
        supported_providers: list[str],
        llm_scopes: list[LLMScope] | None = None,
        *,
        strict: bool = False,
    ) -> None:
        """Validate simulation usage and raise errors if invalid.

        Args:
            llm_calls: LLM calls recorded during test execution.
            supported_providers: Providers that support HTTP simulation.
            llm_scopes: LLMScope spans from @link_llm decorator.
            strict: If True, raises error. If False (default), emits warning.

        Raises:
            TenroMissingLLMCallError: `@link_llm` ran but no HTTP call was made.
            TenroUnusedSimulationError: If strict=True and simulation was never
                triggered.
        """
        llm_scopes = llm_scopes or []
        self._check_unused_providers(llm_scopes, supported_providers, strict=strict)
        self._check_missing_http_calls(llm_calls, llm_scopes, supported_providers)

    def _check_unused_providers(
        self,
        llm_scopes: list[LLMScope],
        supported_providers: list[str],
        *,
        strict: bool = False,
    ) -> None:
        """Check for completely untriggered providers."""
        not_triggered = self._registered - self._triggered - self._optional
        if not not_triggered:
            return

        provider = sorted(not_triggered)[0]

        scope_calls = [s for s in llm_scopes if s.provider == provider or s.provider is None]
        if scope_calls:
            msg = self._build_missing_call_message(
                provider=provider,
                scopes=scope_calls,
                supported_providers=supported_providers,
            )
            raise TenroMissingLLMCallError(msg)

        msg = self._build_unused_simulation_message(
            provider=provider,
            supported_providers=supported_providers,
        )
        if strict:
            raise TenroUnusedSimulationError(msg)
        else:
            warnings.warn(msg, TenroUnusedSimulationWarning, stacklevel=4)

    def _check_missing_http_calls(
        self,
        llm_calls: list[LLMCall],
        llm_scopes: list[LLMScope],
        supported_providers: list[str],
    ) -> None:
        """Check for @link_llm calls that didn't make HTTP requests."""
        covered_scope_ids = {c.llm_scope_id for c in llm_calls if c.llm_scope_id}

        for provider in self._registered - self._optional:
            provider_scopes = [
                s for s in llm_scopes if s.provider == provider or s.provider is None
            ]
            missing_scopes = [s for s in provider_scopes if s.span_id not in covered_scope_ids]

            if missing_scopes:
                msg = self._build_missing_call_message(
                    provider=provider,
                    scopes=missing_scopes,
                    supported_providers=supported_providers,
                )
                raise TenroMissingLLMCallError(msg)

    def _build_missing_call_message(
        self,
        provider: str,
        scopes: list[LLMScope],
        supported_providers: list[str],
    ) -> str:
        """Build error message when @link_llm was used but no HTTP call was made."""
        scope = scopes[0] if scopes else None
        scope_provider = scope.provider if scope else None
        decorator = f"@link_llm('{scope_provider}')" if scope_provider else "@link_llm"

        func_name = scope.caller_name if scope else None
        func_display = f" '{func_name}'" if func_name else ""

        lines = [
            f"{decorator}{func_display} was called but no LLM HTTP request was made",
            "",
            "Your decorated function returned without hitting the LLM provider API.",
            "This breaks the harness because the simulation never had a chance to activate.",
            "",
            f"  Simulated provider: {provider}",
            f"  @link_llm calls observed: {len(scopes)}",
            "  LLM HTTP calls intercepted: 0",
        ]

        if scopes:
            lines.append("")
            lines.append("Functions that didn't make HTTP calls:")
            seen: set[str] = set()
            for scope in scopes:
                if scope.caller_signature and scope.caller_signature not in seen:
                    seen.add(scope.caller_signature)
                    loc = f" at {scope.caller_location}" if scope.caller_location else ""
                    lines.append(f"  - {scope.caller_signature}{loc}")

        lines.extend(
            [
                "",
                "This usually means your @link_llm function returned a stub",
                "instead of calling the real LLM client.",
                "",
                f"Supported HTTP providers: {', '.join(sorted(supported_providers))}",
                "Note: HTTP interception only works with httpx-based clients "
                "(e.g., openai, anthropic, google-genai).",
            ]
        )

        example = ProviderSchemaFactory.get_code_example(provider)
        if example:
            lines.extend(
                [
                    "",
                    "Fix: Ensure your decorated function makes real HTTP calls:",
                    f"  @link_llm('{provider}')",
                    "  def call_llm(prompt: str) -> str:",
                    f"      {example['client']}",
                    f"      {example['call']}",
                    f"      {example['extract']}",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Fix: Ensure your decorated function makes real HTTP calls",
                    f"to the {provider} API.",
                ]
            )

        return "\n".join(lines)

    def _build_unused_simulation_message(
        self,
        provider: str,
        supported_providers: list[str],
    ) -> str:
        """Build error message for unused simulation."""
        lines = [
            f"HTTP simulation for '{provider}' was never triggered",
            "",
            "Your test configured simulate_llm() for this provider,",
            "but no LLM calls were recorded for it.",
            "",
            f"  Provider: {provider}",
            "  @link_llm calls recorded: 0",
            "  HTTP calls intercepted: 0",
            "",
            "This usually means the code path that would call the LLM",
            "never executed during the test.",
            "",
            f"Supported LLM providers: {', '.join(sorted(supported_providers))}",
            "",
            "Fix: Remove the unused simulation or execute the code path:",
            f"  construct.simulate_llm('{provider}', response='...')",
            "  # ... trigger the LLM call ...",
        ]
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all tracking state."""
        self._registered.clear()
        self._optional.clear()
        self._triggered.clear()
