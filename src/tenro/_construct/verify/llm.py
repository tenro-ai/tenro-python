# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""LLM verification methods for construct testing.

Provides 3 core verification methods for LLM calls:
- verify_llm() - flexible verification with output checking and where selector
- verify_llm_never() - explicit "never called" check
- verify_llms() - aggregate/range queries
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from tenro._construct.verify.engine import verify_call_count
from tenro._construct.verify.output import (
    verify_output,
    verify_output_contains,
    verify_output_exact,
)
from tenro.providers import Provider

if TYPE_CHECKING:
    from tenro._core.spans import AgentRun, LLMCall


class _LLMValueWrapper:
    """Wrapper to make LLM values work with verify_output functions."""

    def __init__(self, value: Any) -> None:
        self.value = value


class LLMVerifications:
    """LLM verification methods."""

    def __init__(self, llm_calls: list[LLMCall], agent_runs: list[AgentRun]) -> None:
        """Initialize with LLM calls and agent runs."""
        self._llm_calls = llm_calls
        self._agent_runs = agent_runs

    @staticmethod
    def _normalize_provider_filter(provider: Provider | str | None) -> str | None:
        """Normalize provider filter for matching against stored provider IDs.

        Defensive normalization in case this layer is called directly.
        Construct layer handles validation; this just normalizes for matching.

        Args:
            provider: Provider enum, string, or None.

        Returns:
            Normalized provider string, or None if no filter.
        """
        if provider is None:
            return None
        if isinstance(provider, Provider):
            return provider.value
        normalized = provider.strip().lower()
        return normalized if normalized else None

    def verify_llm(
        self,
        provider: Provider | str | None = None,
        *,
        target: str | None = None,
        count: int | None = None,
        output: Any = None,
        output_contains: str | None = None,
        output_exact: Any = None,
        where: str | None = None,
        call_index: int | None = None,
    ) -> None:
        """Verify LLM was called with optional output checking.

        Args:
            provider: Optional provider filter. Use Provider.OPENAI for built-ins.
            target: Optional target filter (e.g., `openai.chat.completions.create`).
            count: Expected number of calls (`None` = at least once).
            output: Expected output (dict=subset match, scalar=exact).
            output_contains: Expected substring in output (strings only).
            output_exact: Expected output (strict deep equality).
            where: Selector for which part to check:
                - `None` (default): response text
                - `"json"`: parse response as JSON
                - `"model"`: model name
            call_index: Which call to check (0=first, -1=last, None=any).

        Examples:
            >>> construct.verify_llm()  # at least once
            >>> construct.verify_llm(Provider.OPENAI, count=1)
            >>> construct.verify_llm(output_contains="success")
            >>> construct.verify_llm(where="json", output={"temp": 72})
            >>> construct.verify_llm(where="model", output="gpt-5")
        """
        # Validate target is not a Provider enum (common mistake)
        if isinstance(target, Provider):
            raise TypeError(
                f"target= expects a string path, not Provider.{target.name}. "
                f"Use provider={target!r} instead."
            )
        provider_filter = self._normalize_provider_filter(provider)

        if count == 0:
            self.verify_llm_never(provider_filter, target=target)
            return

        calls = self._llm_calls
        if provider_filter:
            calls = [c for c in calls if c.provider == provider_filter]

        # Output verification
        if output is not None or output_contains is not None or output_exact is not None:
            items = self._apply_where_selector(calls, where)

            if output is not None:
                verify_output(items, "value", output, "LLM", call_index=call_index)
            elif output_contains is not None:
                verify_output_contains(
                    items, "value", output_contains, "LLM", call_index=call_index
                )
            elif output_exact is not None:
                verify_output_exact(items, "value", output_exact, "LLM", call_index=call_index)
            return

        verify_call_count(
            calls=self._llm_calls,
            agent_runs=self._agent_runs,
            count=count,
            min=1 if count is None else None,
            max=None,
            name_filter=target,
            agent_filter=None,
            event_type="llm",
            provider_filter=provider_filter,
        )

    def _apply_where_selector(
        self, calls: list[LLMCall], where: str | None
    ) -> list[_LLMValueWrapper]:
        """Apply where selector to extract values from LLM calls."""
        result = []
        for call in calls:
            if where is None:
                result.append(_LLMValueWrapper(call.response))
            elif where == "json":
                try:
                    parsed = json.loads(call.response or "{}")
                    result.append(_LLMValueWrapper(parsed))
                except json.JSONDecodeError:
                    result.append(_LLMValueWrapper(None))
            elif where == "model":
                result.append(_LLMValueWrapper(call.model))
            else:
                msg = f"Unknown where selector: {where!r}. "
                msg += "Supported: None, 'json', 'model'"
                raise ValueError(msg)
        return result

    def verify_llm_never(
        self,
        provider: Provider | str | None = None,
        *,
        target: str | None = None,
    ) -> None:
        """Verify LLM was never called.

        Args:
            provider: Optional provider filter. Use Provider.OPENAI for built-ins.
            target: Optional target filter (e.g., `"openai.chat.completions.create"`).

        Raises:
            AssertionError: If LLM was called.

        Examples:
            >>> construct.verify_llm_never()  # no LLM calls at all
            >>> construct.verify_llm_never(Provider.ANTHROPIC)
        """
        # Validate target is not a Provider enum (common mistake)
        if isinstance(target, Provider):
            raise TypeError(
                f"target= expects a string path, not Provider.{target.name}. "
                f"Use provider={target!r} instead."
            )
        provider_filter = self._normalize_provider_filter(provider)
        verify_call_count(
            calls=self._llm_calls,
            agent_runs=self._agent_runs,
            count=0,
            min=None,
            max=None,
            name_filter=target,
            agent_filter=None,
            event_type="llm",
            provider_filter=provider_filter,
        )

    def verify_llms(
        self,
        provider: Provider | str | None = None,
        *,
        count: int | None = None,
        min: int | None = None,
        max: int | None = None,
        target: str | None = None,
    ) -> None:
        """Verify LLM calls with optional count/range and target/provider filter.

        Args:
            provider: Optional provider filter. Use Provider.OPENAI for built-ins.
            count: Expected exact number of calls (mutually exclusive with min/max).
            min: Minimum number of calls (inclusive).
            max: Maximum number of calls (inclusive).
            target: Optional target filter.

        Raises:
            AssertionError: If verification fails.
            ValueError: If `count` and min/max are both specified.

        Examples:
            >>> construct.verify_llms()  # at least one LLM call
            >>> construct.verify_llms(count=2)  # exactly 2 LLM calls
            >>> construct.verify_llms(Provider.OPENAI, min=1, max=3)
        """
        # Validate target is not a Provider enum (common mistake)
        if isinstance(target, Provider):
            raise TypeError(
                f"target= expects a string path, not Provider.{target.name}. "
                f"Use provider={target!r} instead."
            )
        provider_filter = self._normalize_provider_filter(provider)
        verify_call_count(
            calls=self._llm_calls,
            agent_runs=self._agent_runs,
            count=count,
            min=min,
            max=max,
            name_filter=target,
            agent_filter=None,
            event_type="llm",
            provider_filter=provider_filter,
        )


__all__ = ["LLMVerifications"]
