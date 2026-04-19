# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pytest report annotation for wrapped Tenro blocks.

Provider SDKs (OpenAI, Anthropic) re-raise transport-layer exceptions
as their own ``APIConnectionError`` types. When a Tenro block is buried
under such a wrapper, the default failure report surfaces only the
generic ``Connection error.`` text. The hook in this module detects
the wrapped block and appends a labelled section to the failure report
so the underlying cause is visible without inspecting ``__cause__`` by
hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tenro.errors import TenroUnexpectedLLMCallError

if TYPE_CHECKING:
    from collections.abc import Generator


# Provider SDK exception types known to wrap transport-layer exceptions.
# Match by (module_prefix, class_name) so a user-defined class with the
# same name does not accidentally trigger annotation.
_KNOWN_SDK_WRAPPERS: frozenset[tuple[str, str]] = frozenset(
    {
        ("openai.", "APIConnectionError"),
        ("openai.", "APITimeoutError"),
        ("anthropic.", "APIConnectionError"),
        ("anthropic.", "APITimeoutError"),
    }
)


def _is_known_sdk_wrapper(exc: BaseException) -> bool:
    """Return True if ``exc`` is a known provider-SDK transport wrapper."""
    cls = type(exc)
    name = cls.__name__
    module = (cls.__module__ or "") + "."
    return any(
        name == wrapper_name and module.startswith(prefix)
        for prefix, wrapper_name in _KNOWN_SDK_WRAPPERS
    )


def _find_attributable_block(
    exc: BaseException,
) -> TenroUnexpectedLLMCallError | None:
    """Find a Tenro block strictly attributable to ``exc``.

    Walks ``__cause__`` only — never ``__context__``, which would
    mis-attribute an assertion raised inside ``except`` to a previously
    handled block.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TenroUnexpectedLLMCallError):
            return current
        if isinstance(current, BaseExceptionGroup):
            return _attribute_group(current)
        current = current.__cause__
    return None


def _attribute_group(
    group: BaseExceptionGroup[BaseException],
) -> TenroUnexpectedLLMCallError | None:
    """Attribute a Tenro block to a group only if every member resolves to one."""
    attributable = [_find_attributable_block(member) for member in group.exceptions]
    real = [b for b in attributable if b is not None]
    if real and len(real) == len(attributable):
        return real[0]
    return None


def _find_block_for_annotation(
    top_exc: BaseException,
) -> TenroUnexpectedLLMCallError | None:
    """Find a Tenro block attributable to ``top_exc``, or ``None``.

    Annotates only the block itself, a known SDK transport wrapper, or
    a group whose members all attribute. Mixed groups return ``None``
    to avoid mislabeling an unrelated sibling failure.
    """
    if isinstance(top_exc, TenroUnexpectedLLMCallError):
        return top_exc
    if isinstance(top_exc, BaseExceptionGroup):
        return _attribute_top_level_group(top_exc)
    if not _is_known_sdk_wrapper(top_exc):
        return None
    return _find_attributable_block(top_exc)


def _attribute_top_level_group(
    group: BaseExceptionGroup[BaseException],
) -> TenroUnexpectedLLMCallError | None:
    """Attribute a top-level group only when every member resolves to a block."""
    resolved = [
        _find_block_for_annotation(member)
        for member in group.exceptions
        if isinstance(member, BaseException)
    ]
    if resolved and all(b is not None for b in resolved):
        return resolved[0]
    return None


def _build_block_section(
    top_exc: BaseException,
    blocked: TenroUnexpectedLLMCallError,
) -> str:
    """Compose the report section text for a Tenro block."""
    from tenro._block_message import resolution_guidance

    top_type = type(top_exc).__name__
    # Use the blocked domain (not blocked.url) so paths and query params
    # never reach pytest/JUnit/HTML report artifacts.
    if top_exc is blocked or not _is_known_sdk_wrapper(top_exc):
        intro = f"Tenro blocked a real call to {blocked.domain} (no simulation set up)."
    else:
        intro = (
            f"That {top_type} above is Tenro stopping a real call to "
            f"{blocked.domain} (no simulation set up)."
        )
    return f"{intro}\n\n{resolution_guidance(blocked.domain)}"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, None, None]:
    """Append a Tenro block diagnostic to failure reports."""
    outcome = yield
    report = outcome.get_result()  # type: ignore[attr-defined]
    if call.excinfo is None or not getattr(report, "failed", False):
        return

    blocked = _find_block_for_annotation(call.excinfo.value)
    if blocked is None:
        return

    section_content = _build_block_section(call.excinfo.value, blocked)
    # Mutate report.sections directly: item.add_report_section() runs too
    # late — the TestReport was already built from item._report_sections.
    report.sections.append(("Captured tenro " + call.when, section_content))
