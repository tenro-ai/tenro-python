# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""ContextVars for linking decorators."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tenro._construct.construct import Construct

_construct_stack: ContextVar[tuple[Construct, ...]] = ContextVar("construct_stack", default=())


@dataclass(frozen=True)
class GuardKey:
    """Key for re-entrancy guard."""

    kind: str
    target_id: int


_active_guards: ContextVar[frozenset[GuardKey]] = ContextVar("tenro_guards", default=frozenset())


def guard_enter(key: GuardKey) -> Token[frozenset[GuardKey]] | None:
    """Attempt to acquire re-entrancy guard.

    Args:
        key: Guard key identifying the re-entrancy boundary.

    Returns:
        Token if guard acquired (caller should create span).
        None if already active (re-entry, caller should pass-through).
    """
    guards = _active_guards.get()
    if key in guards:
        return None
    return _active_guards.set(guards | {key})


def guard_exit(token: Token[frozenset[GuardKey]] | None) -> None:
    """Release a re-entrancy guard.

    Args:
        token: Guard token from `guard_enter`, or None if guard was not acquired.
    """
    if token is not None:
        try:  # noqa: SIM105
            _active_guards.reset(token)
        except ValueError:
            pass


def get_active_construct() -> Construct | None:
    """Get the currently active construct from context."""
    stack = _construct_stack.get()
    return stack[-1] if stack else None


def get_construct_stack() -> tuple[Construct, ...]:
    """Get the full construct stack."""
    return _construct_stack.get()


def push_construct(construct: Construct) -> None:
    """Push a construct onto the stack."""
    stack = _construct_stack.get()
    _construct_stack.set((*stack, construct))


def pop_construct() -> Construct | None:
    """Pop a construct from the stack.

    Returns:
        The popped construct, or None if stack was empty.
    """
    stack = _construct_stack.get()
    if not stack:
        return None
    _construct_stack.set(stack[:-1])
    return stack[-1]


def set_active_construct(construct: Construct | None) -> None:
    """Set the active construct in context."""
    if construct is None:
        pop_construct()
    else:
        push_construct(construct)


__all__ = [
    "GuardKey",
    "get_active_construct",
    "get_construct_stack",
    "guard_enter",
    "guard_exit",
    "pop_construct",
    "push_construct",
    "set_active_construct",
]
