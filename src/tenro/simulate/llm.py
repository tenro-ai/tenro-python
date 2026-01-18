# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""LLM simulation API.

- llm.verify()                  # at least once (1+)
- llm.verify_many(count=1)      # exactly once
- llm.verify_many(count=2)      # exactly twice
- llm.verify_never()            # never called

Examples:
    >>> from tenro.simulate import llm
    >>> from tenro import Provider
    >>>
    >>> def test_my_agent(construct):
    ...     llm.simulate(Provider.OPENAI, response="Hello!")
    ...     my_agent.run()
    ...     llm.verify(Provider.OPENAI)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tenro.providers import Provider
from tenro.simulate._helpers import get_construct_or_raise

if TYPE_CHECKING:
    from collections.abc import Callable

    from tenro.spans import LLMCallSpan


def simulate(
    provider: Provider | str | None = None,
    *,
    target: str | Callable[..., Any] | None = None,
    response: str | None = None,
    responses: str | list[str | Exception | dict[str, Any]] | None = None,
    model: str | None = None,
    tool_calls: list[Any] | None = None,
    use_http: bool | None = None,
    optional: bool = False,
    **response_kwargs: Any,
) -> None:
    """Register an LLM simulation."""
    get_construct_or_raise().simulate_llm(
        provider,
        target=target,
        response=response,
        responses=responses,
        model=model,
        tool_calls=tool_calls,
        use_http=use_http,
        optional=optional,
        **response_kwargs,
    )


def verify(
    provider: Provider | str | None = None,
    *,
    target: str | Callable[..., Any] | None = None,
    output: Any = None,
    output_contains: str | None = None,
    output_exact: Any = None,
    where: Callable[[LLMCallSpan], bool] | None = None,
    call_index: int | None = None,
) -> LLMCallSpan | None:
    """Verify an LLM was called at least once (1+).

    Use verify_many(count=N) for exact count verification.

    Args:
        provider: Provider to verify (optional).
        target: LLM function or dotted path to verify.
        output: Expected output value.
        output_contains: Substring that should appear in output.
        output_exact: Exact expected output.
        where: Predicate filter for span selection.
        call_index: Index of specific call to verify.

    Returns:
        The matching LLM call span, or None if no calls made.
    """
    if callable(target) and not isinstance(target, str):
        from tenro._construct.simulate.target_resolution import resolve_all_target_paths

        paths = resolve_all_target_paths(target)
        resolved_target = paths[0] if paths else None
    else:
        resolved_target = target

    get_construct_or_raise().verify_llm(
        provider,
        target=resolved_target,
        output=output,
        output_contains=output_contains,
        output_exact=output_exact,
        call_index=call_index,
    )
    all_calls = calls()
    if not all_calls:
        return None
    filtered = list(all_calls)
    if provider:
        provider_str = provider.value if isinstance(provider, Provider) else provider
        filtered = [c for c in filtered if c.provider == provider_str]
    if where:
        filtered = [c for c in filtered if where(c)]
    if call_index is not None:
        if 0 <= call_index < len(filtered) or (call_index < 0 and abs(call_index) <= len(filtered)):
            return filtered[call_index]
        return None
    return filtered[-1] if filtered else None


def verify_never(
    provider: Provider | str | None = None,
    *,
    target: str | None = None,
) -> None:
    """Verify an LLM was never called."""
    get_construct_or_raise().verify_llm_never(provider, target=target)


def verify_many(
    provider: Provider | str | None = None,
    *,
    target: str | Callable[..., Any] | None = None,
    where: Callable[[LLMCallSpan], bool] | None = None,
    count: int | None = None,
    at_least: int | None = None,
    at_most: int | None = None,
) -> tuple[LLMCallSpan, ...]:
    """Verify LLM call count.

    Args:
        provider: Provider to verify (optional).
        target: LLM function or dotted path to verify.
        where: Predicate filter for span selection.
        count: Exact number of calls expected.
        at_least: Minimum number of calls expected.
        at_most: Maximum number of calls expected.

    Examples:
        >>> llm.verify_many(count=1)      # exactly once
        >>> llm.verify_many(count=2)      # exactly twice
        >>> llm.verify_many(at_least=1)   # 1 or more
    """
    # Resolve callable target to string path
    resolved_target: str | None = None
    if callable(target) and not isinstance(target, str):
        from tenro._construct.simulate.target_resolution import resolve_all_target_paths

        paths = resolve_all_target_paths(target)
        resolved_target = paths[0] if paths else None
    else:
        resolved_target = target

    get_construct_or_raise().verify_llms(
        provider,
        count=count,
        min=at_least,
        max=at_most,
        target=resolved_target,
    )
    all_calls = calls()
    if where:
        return tuple(c for c in all_calls if where(c))
    return all_calls


def calls() -> tuple[LLMCallSpan, ...]:
    """Get all LLM calls (read-only access)."""
    return tuple(get_construct_or_raise().llm_calls)


__all__ = ["calls", "simulate", "verify", "verify_many", "verify_never"]
