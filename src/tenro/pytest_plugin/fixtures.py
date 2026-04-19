# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for Tenro SDK.

This module provides fixtures for:
- Construct: Simulate and intercept tool/LLM calls
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tenro._construct.construct import Construct


@pytest.fixture
def construct(request: pytest.FixtureRequest) -> Generator[Construct, None, None]:
    """Provide a Construct instance for testing.

    This is the primary pytest fixture for Tenro testing. The construct
    automatically handles setup/teardown of all patches and simulations.

    Options set via ``@tenro.simulate(...)`` marker are forwarded to the
    Construct constructor, enabling per-test control::

        @tenro.simulate(allow_real_llm_calls=True)
        def test_live_integration(construct):
            ...

    Supported marker kwargs: ``allow_real_llm_calls``, ``fail_unused``.

    Args:
        request: Pytest fixture request for accessing test item.

    Yields:
        Construct instance ready for simulation and testing.

    Examples:
        >>> def test_my_agent(construct: Construct):
        ...     construct.simulate_llm(Provider.OPENAI, response="Hi")
        ...     result = my_agent.run()
        ...     assert "Hi" in result

        Multi-turn conversations:
        >>> def test_conversation(construct: Construct):
        ...     construct.simulate_llm(
        ...         Provider.ANTHROPIC,
        ...         responses=["Turn 1", "Turn 2", "Turn 3"]
        ...     )
        ...     agent = ConversationAgent()
        ...     agent.run()
    """
    from tenro._construct.construct import Construct
    from tenro.pytest_plugin.construct_registry import register_construct
    from tenro.pytest_plugin.plugin import is_fail_unused

    marker_kwargs = _collect_marker_kwargs(request)
    construct_kwargs: dict[str, Any] = {
        "fail_unused": marker_kwargs.get("fail_unused", is_fail_unused()),
    }
    if "allow_real_llm_calls" in marker_kwargs:
        construct_kwargs["allow_real_llm_calls"] = marker_kwargs["allow_real_llm_calls"]

    c = Construct(**construct_kwargs)
    register_construct(c)
    request.node._tenro_construct = c  # pyright: ignore[reportAttributeAccessIssue]
    try:
        with c:
            yield c
    finally:
        _release_construct(request, c)


def _release_construct(request: pytest.FixtureRequest, c: Construct) -> None:
    """Drop the Construct's registry entry and the strong reference on ``request.node``.

    pytest keeps every test item for the whole session, so leaving the
    strong reference on ``request.node`` would pin the Construct until
    session end.
    """
    from tenro.pytest_plugin.construct_registry import unregister_construct

    unregister_construct(c)
    with contextlib.suppress(AttributeError):
        del request.node._tenro_construct


# Accepted marker kwargs and their expected types
_MARKER_SCHEMA: dict[str, type | tuple[type, ...]] = {
    "allow_real_llm_calls": bool,
    "fail_unused": bool,
}


def _collect_marker_kwargs(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Merge ``tenro`` marker kwargs from all scopes and validate types.

    Iterates markers from nearest (method) to outermost (module).
    Nearest scope wins per key. Rejects unknown keys and wrong types.

    Args:
        request: Pytest fixture request.

    Returns:
        Merged and validated kwargs dict.

    Raises:
        ValueError: If an unknown kwarg or wrong type is found.
    """
    merged: dict[str, Any] = {}
    for marker in request.node.iter_markers("tenro"):
        for key, value in marker.kwargs.items():
            if key not in _MARKER_SCHEMA:
                raise ValueError(
                    f"@tenro.simulate() got unexpected keyword argument {key!r}. "
                    f"Supported: {', '.join(sorted(_MARKER_SCHEMA))}"
                )
            # Nearest scope wins — skip if already set
            if key not in merged:
                expected = _MARKER_SCHEMA[key]
                if not isinstance(value, expected):
                    type_name = (
                        expected.__name__
                        if isinstance(expected, type)
                        else " | ".join(t.__name__ for t in expected)
                    )
                    raise ValueError(
                        f"@tenro.simulate({key}=) expected {type_name}, "
                        f"got {type(value).__name__}: {value!r}"
                    )
                merged[key] = value
    return merged
