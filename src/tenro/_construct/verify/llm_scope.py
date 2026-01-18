# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""LLMScope verification methods for construct testing.

Verifies @link_llm decorated function calls (LLMScope spans).
Separate from verify_llm() which verifies HTTP-intercepted LLMCall spans.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from tenro._construct.verify.format import format_error_with_trace
from tenro._construct.verify.output import is_subset_match
from tenro.errors import TenroVerificationError
from tenro.util.list_helpers import normalize_and_validate_index

if TYPE_CHECKING:
    from tenro._core.spans import LLMScope


class LLMScopeVerifications:
    """LLMScope verification methods."""

    def __init__(self, llm_scopes: list[LLMScope]) -> None:
        """Initialize with LLMScope spans.

        Args:
            llm_scopes: List of LLMScope spans from @link_llm decorator.
        """
        self._llm_scopes = llm_scopes

    def verify_llm_scope(
        self,
        target: Callable[..., Any] | None = None,
        *,
        times: int | None = None,
        scope_index: int | None = None,
        input: tuple[tuple[Any, ...], dict[str, Any]] | Any = None,
        input_contains: str | None = None,
        input_exact: Any = None,
        output: Any = None,
        output_contains: str | None = None,
        output_exact: Any = None,
    ) -> None:
        """Verify @link_llm function calls (LLMScope).

        Use this to assert what your code saw and returned:
        - input: the function's (args_tuple, kwargs_dict) pair
        - output: the function's return value (after post-processing)

        Args:
            target: Function reference to filter by (e.g., `extract_entities`).
            times: Expected number of calls. Mutually exclusive with scope_index.
            scope_index: Which call to check (0=first, -1=last). Exclusive
                with times.
            input: Expected input (smart match on (args, kwargs) pair).
            input_contains: Expected substring in input repr.
            input_exact: Expected input (strict equality).
            output: Expected output (dict=subset match, scalar=exact).
            output_contains: Expected substring in output repr.
            output_exact: Expected output (strict deep equality).

        Raises:
            AssertionError: If verification fails.
            TypeError: If mutually exclusive parameters are combined.

        Examples:
            >>> construct.verify_llm_scope(input_exact=(("hello",), {"lang": "en"}))
            >>> construct.verify_llm_scope(input_contains="quick brown fox")
            >>> construct.verify_llm_scope(output={"status": "ok"})
            >>> construct.verify_llm_scope(output_exact=["fox", "dog"])
            >>> construct.verify_llm_scope(target=extract_entities, times=1)
        """
        self._validate_mutual_exclusivity(times, scope_index, "times", "scope_index")
        self._validate_mutual_exclusivity(input, input_contains, "input", "input_contains")
        self._validate_mutual_exclusivity(input, input_exact, "input", "input_exact")
        self._validate_mutual_exclusivity(
            input_contains, input_exact, "input_contains", "input_exact"
        )
        self._validate_mutual_exclusivity(output, output_contains, "output", "output_contains")
        self._validate_mutual_exclusivity(output, output_exact, "output", "output_exact")
        self._validate_mutual_exclusivity(
            output_contains, output_exact, "output_contains", "output_exact"
        )

        scopes = self._filter_by_target(target)

        if times is not None:
            self._verify_times(scopes, times, target)
            return

        if output is not None:
            self._verify_output(scopes, output, is_subset_match, scope_index)
            return
        if output_contains is not None:
            self._verify_output_contains(scopes, output_contains, scope_index)
            return
        if output_exact is not None:
            self._verify_output(scopes, output_exact, lambda a, e: a == e, scope_index)
            return

        if input is not None:
            self._verify_input(scopes, input, is_subset_match, scope_index)
            return
        if input_contains is not None:
            self._verify_input_contains(scopes, input_contains, scope_index)
            return
        if input_exact is not None:
            self._verify_input(scopes, input_exact, lambda a, e: a == e, scope_index)
            return

        # Default: verify at least one scope exists
        if not scopes:
            target_name = target.__name__ if target else "any"
            raise TenroVerificationError(f"No @link_llm calls found for {target_name}")

    def _validate_mutual_exclusivity(
        self,
        param1: Any,
        param2: Any,
        name1: str,
        name2: str,
    ) -> None:
        """Raise TypeError if both parameters are provided."""
        if param1 is not None and param2 is not None:
            raise TypeError(f"{name1} and {name2} are mutually exclusive")

    def _filter_by_target(self, target: Callable[..., Any] | None) -> list[LLMScope]:
        """Filter scopes by target function name."""
        if target is None:
            return self._llm_scopes
        target_name = target.__name__
        return [s for s in self._llm_scopes if s.caller_name == target_name]

    def _verify_times(
        self,
        scopes: list[LLMScope],
        times: int,
        target: Callable[..., Any] | None,
    ) -> None:
        """Verify exact call count."""
        actual = len(scopes)
        if actual != times:
            target_name = target.__name__ if target else "@link_llm"
            msg = f"Expected {target_name} to be called {times} time(s), "
            msg += f"but was called {actual} time(s)"
            raise TenroVerificationError(msg)

    def _verify_output(
        self,
        scopes: list[LLMScope],
        expected: Any,
        matcher: Callable[[Any, Any], bool],
        scope_index: int | None,
    ) -> None:
        """Verify output using matcher function."""
        if not scopes:
            raise TenroVerificationError("No @link_llm calls found")

        if scope_index is not None:
            scope, _ = normalize_and_validate_index(scopes, scope_index, "llm_scope")
            actual = scope.output_data
            if matcher(actual, expected):
                return
            header = f"llm_scope[{scope_index}] output mismatch:"
            msg = format_error_with_trace(header, expected, actual, span=scope)
            raise TenroVerificationError(msg)

        for scope in scopes:
            if matcher(scope.output_data, expected):
                return
        actuals = [s.output_data for s in scopes]
        header = f"llm_scope output mismatch (checked all {len(scopes)} calls):"
        msg = format_error_with_trace(header, expected, actuals, span=None)
        raise TenroVerificationError(msg)

    def _verify_output_contains(
        self,
        scopes: list[LLMScope],
        expected_text: str,
        scope_index: int | None,
    ) -> None:
        """Verify output contains expected substring (using repr)."""
        if not scopes:
            raise TenroVerificationError("No @link_llm calls found")

        def contains_text(output: Any) -> bool:
            return expected_text in repr(output)

        if scope_index is not None:
            scope, _ = normalize_and_validate_index(scopes, scope_index, "llm_scope")
            if contains_text(scope.output_data):
                return
            header = f"llm_scope[{scope_index}] output does not contain '{expected_text}':"
            actual = repr(scope.output_data)
            msg = format_error_with_trace(header, expected_text, actual, span=scope)
            raise TenroVerificationError(msg)

        for scope in scopes:
            if contains_text(scope.output_data):
                return
        actuals = [repr(s.output_data) for s in scopes]
        header = (
            f"llm_scope output does not contain '{expected_text}' "
            f"(checked all {len(scopes)} calls):"
        )
        msg = format_error_with_trace(header, expected_text, actuals, span=None)
        raise TenroVerificationError(msg)

    def _verify_input(
        self,
        scopes: list[LLMScope],
        expected: Any,
        matcher: Callable[[Any, Any], bool],
        scope_index: int | None,
    ) -> None:
        """Verify input using matcher function."""
        if not scopes:
            raise TenroVerificationError("No @link_llm calls found")

        def get_input(scope: LLMScope) -> tuple[tuple[Any, ...], dict[str, Any]]:
            return (scope.input_data, scope.input_kwargs)

        if scope_index is not None:
            scope, _ = normalize_and_validate_index(scopes, scope_index, "llm_scope")
            actual = get_input(scope)
            if matcher(actual, expected):
                return
            header = f"llm_scope[{scope_index}] input mismatch:"
            msg = format_error_with_trace(header, expected, actual, span=scope)
            raise TenroVerificationError(msg)

        for scope in scopes:
            if matcher(get_input(scope), expected):
                return
        actuals = [get_input(s) for s in scopes]
        header = f"llm_scope input mismatch (checked all {len(scopes)} calls):"
        msg = format_error_with_trace(header, expected, actuals, span=None)
        raise TenroVerificationError(msg)

    def _verify_input_contains(
        self,
        scopes: list[LLMScope],
        expected_text: str,
        scope_index: int | None,
    ) -> None:
        """Verify input contains expected substring (using repr)."""
        if not scopes:
            raise TenroVerificationError("No @link_llm calls found")

        def get_input_repr(scope: LLMScope) -> str:
            return repr((scope.input_data, scope.input_kwargs))

        def contains_text(scope: LLMScope) -> bool:
            return expected_text in get_input_repr(scope)

        if scope_index is not None:
            scope, _ = normalize_and_validate_index(scopes, scope_index, "llm_scope")
            if contains_text(scope):
                return
            header = f"llm_scope[{scope_index}] input does not contain '{expected_text}':"
            msg = format_error_with_trace(header, expected_text, get_input_repr(scope), span=scope)
            raise TenroVerificationError(msg)

        for scope in scopes:
            if contains_text(scope):
                return
        actuals = [get_input_repr(s) for s in scopes]
        header = (
            f"llm_scope input does not contain '{expected_text}' (checked all {len(scopes)} calls):"
        )
        msg = format_error_with_trace(header, expected_text, actuals, span=None)
        raise TenroVerificationError(msg)


__all__ = ["LLMScopeVerifications"]
