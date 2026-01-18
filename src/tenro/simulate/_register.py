# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Runtime registration for capture-safe simulation of undecorated callables.

This module provides `register()` for opting-in undecorated callables
to capture-safe simulation via __code__ patching.

Registration enables simulation of third-party functions you don't control,
but requires explicit opt-in due to global side effects.
"""

from __future__ import annotations

import inspect
import platform
import types
import warnings
import weakref
from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from tenro.errors.simulation import TenroSimulationSetupError

F = TypeVar("F", bound=Callable[..., Any])

# Registry of registered callables: original -> registration info
_registered: weakref.WeakKeyDictionary[Callable[..., Any], RegistrationInfo] = (
    weakref.WeakKeyDictionary()
)


@dataclass
class _GlobalsState:
    """Tracks dispatch function injection per globals dict."""

    refcount: int
    originals: dict[str, Any]


@dataclass
class _RegisteredTarget:
    """Stored state for a patched function."""

    original_code: types.CodeType
    original_defaults: tuple[Any, ...] | None
    original_kwdefaults: dict[str, Any] | None
    func_globals: dict[str, Any]
    canonical_key: str
    kind: Literal["sync", "async", "gen", "asyncgen"]


# Registry mapping func_id -> original state
_REGISTRY: dict[int, _RegisteredTarget] = {}
_next_id: int = 0

# Patched functions: id(func) -> func_id
_patched: dict[int, int] = {}

# Per-globals tracking: id(func.__globals__) -> state
_globals_state: dict[int, _GlobalsState] = {}


class RegistrationInfo:
    """Information about a registered callable."""

    __slots__ = ("failure_reason", "is_patchable", "target_path")

    def __init__(
        self,
        target_path: str,
        is_patchable: bool,
        failure_reason: str | None = None,
    ) -> None:
        self.target_path = target_path
        self.is_patchable = is_patchable
        self.failure_reason = failure_reason


def _tenro_dispatch_sync(func_id: int, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Dispatch sync function call through simulation registry."""
    reg = _REGISTRY.get(func_id)
    if reg is None:
        raise RuntimeError("Tenro trampoline called without registration")

    from tenro.linking.dispatch import dispatch_sync

    def call_real(*a: Any, **kw: Any) -> Any:
        fn = types.FunctionType(
            reg.original_code,
            reg.func_globals,
        )
        fn.__defaults__ = reg.original_defaults
        fn.__kwdefaults__ = reg.original_kwdefaults
        return fn(*a, **kw)

    result, _ = dispatch_sync(reg.canonical_key, call_real, args, kwargs)
    return result


async def _tenro_dispatch_async(func_id: int, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Dispatch async function call through simulation registry."""
    reg = _REGISTRY.get(func_id)
    if reg is None:
        raise RuntimeError("Tenro trampoline called without registration")

    from tenro.linking.dispatch import dispatch_async

    async def call_real(*a: Any, **kw: Any) -> Any:
        fn = types.FunctionType(
            reg.original_code,
            reg.func_globals,
        )
        fn.__defaults__ = reg.original_defaults
        fn.__kwdefaults__ = reg.original_kwdefaults
        return await fn(*a, **kw)

    result, _ = await dispatch_async(reg.canonical_key, call_real, args, kwargs)
    return result


def _tenro_dispatch_gen(
    func_id: int, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Generator[Any, None, None]:
    """Dispatch generator function call through simulation registry."""
    reg = _REGISTRY.get(func_id)
    if reg is None:
        raise RuntimeError("Tenro trampoline called without registration")

    from tenro.linking.dispatch import dispatch_gen

    def call_real(*a: Any, **kw: Any) -> Generator[Any, None, None]:
        fn = types.FunctionType(
            reg.original_code,
            reg.func_globals,
        )
        fn.__defaults__ = reg.original_defaults
        fn.__kwdefaults__ = reg.original_kwdefaults
        yield from fn(*a, **kw)

    gen, _ = dispatch_gen(reg.canonical_key, call_real, args, kwargs)
    yield from gen


async def _tenro_dispatch_asyncgen(
    func_id: int, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> AsyncGenerator[Any, None]:
    """Dispatch async generator function call through simulation registry."""
    reg = _REGISTRY.get(func_id)
    if reg is None:
        raise RuntimeError("Tenro trampoline called without registration")

    from tenro.linking.dispatch import dispatch_asyncgen

    async def call_real(*a: Any, **kw: Any) -> AsyncGenerator[Any, None]:
        fn = types.FunctionType(
            reg.original_code,
            reg.func_globals,
        )
        fn.__defaults__ = reg.original_defaults
        fn.__kwdefaults__ = reg.original_kwdefaults
        async for item in fn(*a, **kw):
            yield item

    gen, _ = await dispatch_asyncgen(reg.canonical_key, call_real, args, kwargs)
    async for item in gen:
        yield item


def _trampoline_sync(
    *args: Any,
    __tenro_id__: int = -1,
    **kwargs: Any,
) -> Any:
    """Static sync trampoline template."""
    return _tenro_dispatch_sync(__tenro_id__, args, kwargs)


async def _trampoline_async(
    *args: Any,
    __tenro_id__: int = -1,
    **kwargs: Any,
) -> Any:
    """Static async trampoline template."""
    return await _tenro_dispatch_async(__tenro_id__, args, kwargs)


def _trampoline_gen(
    *args: Any,
    __tenro_id__: int = -1,
    **kwargs: Any,
) -> Generator[Any, None, None]:
    """Static generator trampoline template."""
    yield from _tenro_dispatch_gen(__tenro_id__, args, kwargs)


async def _trampoline_asyncgen(
    *args: Any,
    __tenro_id__: int = -1,
    **kwargs: Any,
) -> AsyncGenerator[Any, None]:
    """Static async generator trampoline template."""
    async for item in _tenro_dispatch_asyncgen(__tenro_id__, args, kwargs):
        yield item


def register(func: F) -> F:
    """Register a callable for capture-safe simulation.

    Enables capture-safe simulation of undecorated functions via __code__
    patching. Use this for third-party functions you cannot decorate.

    Args:
        func: The callable to register. Must be a pure Python function.

    Returns:
        The same callable (unchanged).

    Raises:
        TenroSimulationSetupError: If the callable is not patchable.

    Example:
        >>> from tenro.simulate import register
        >>> from third_party import their_function
        >>> register(their_function)
        >>> # Now simulation works even for captured references
        >>> tool.simulate(their_function, result="simulated")
    """
    is_patchable, reason = check_patchability(func)
    target_path = _get_target_path(func)

    if not is_patchable:
        raise TenroSimulationSetupError.not_patchable(target_path, reason or "Unknown")

    info = RegistrationInfo(target_path=target_path, is_patchable=True)
    _registered[func] = info

    return func


def is_registered(func: Callable[..., Any]) -> bool:
    """Check if a callable is registered for capture-safe simulation."""
    return func in _registered


def get_registration(func: Callable[..., Any]) -> RegistrationInfo | None:
    """Get registration info for a callable."""
    return _registered.get(func)


def check_patchability(func: Callable[..., Any]) -> tuple[bool, str | None]:
    """Check if a callable is patchable for capture-safe simulation.

    A callable is patchable if ALL of the following are true:
    1. Pure Python function (inspect.isfunction)
    2. Has __code__ attribute that can be assigned
    3. No free variables (__code__.co_freevars is empty)
    4. Not a builtin or C-extension
    5. Running on CPython

    Args:
        func: The callable to check.

    Returns:
        Tuple of (is_patchable, failure_reason).
        failure_reason is None if patchable.
    """
    # Must be a function (not method, builtin, or descriptor)
    if not inspect.isfunction(func):
        if inspect.ismethod(func):
            return False, "Bound methods must use __func__ patching"
        if inspect.isbuiltin(func):
            return False, "Builtin functions cannot be patched"
        if inspect.ismethoddescriptor(func):
            return False, "Method descriptors cannot be patched"
        return False, f"Not a function: {type(func).__name__}"

    # Must have __code__ attribute for patching
    if not hasattr(func, "__code__"):
        return False, "No __code__ attribute"

    # No free variables (closures capture external state)
    code = func.__code__
    if code.co_freevars:
        return False, f"Has closure variables: {code.co_freevars}"

    # Not a builtin module function
    if getattr(func, "__module__", None) == "builtins":
        return False, "Builtin function"

    # CPython required for __code__ patching
    if platform.python_implementation() != "CPython":
        return False, f"Requires CPython (found {platform.python_implementation()})"

    return True, None


def _get_target_path(func: Callable[..., Any]) -> str:
    """Get the fully-qualified dotted path for a callable."""
    module = getattr(func, "__module__", "<unknown>")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", "<unknown>"))
    return f"{module}.{qualname}"


def install_trampoline(func: Callable[..., Any], canonical_key: str) -> None:
    """Install the dispatch trampoline for a registered callable.

    Replaces the function's __code__ with a static trampoline that checks
    the simulation registry at call time. Uses __kwdefaults__ to pass
    the function ID without closures.

    Args:
        func: The registered function to patch.
        canonical_key: The target path for registry lookup.
    """
    global _next_id

    if id(func) in _patched:
        return  # Already patched

    if not inspect.isfunction(func):
        raise TypeError(f"Cannot install trampoline on {type(func).__name__}")

    func_id = _next_id
    _next_id += 1

    # Determine function kind and select template
    template: Callable[..., Any]
    if inspect.isasyncgenfunction(func):
        template = _trampoline_asyncgen
        kind: Literal["sync", "async", "gen", "asyncgen"] = "asyncgen"
    elif inspect.isgeneratorfunction(func):
        template = _trampoline_gen
        kind = "gen"
    elif inspect.iscoroutinefunction(func):
        template = _trampoline_async
        kind = "async"
    else:
        template = _trampoline_sync
        kind = "sync"

    # Save original state before any modifications
    original_code = func.__code__
    original_defaults = func.__defaults__
    original_kwdefaults = func.__kwdefaults__

    dispatch_names = (
        "_tenro_dispatch_sync",
        "_tenro_dispatch_async",
        "_tenro_dispatch_gen",
        "_tenro_dispatch_asyncgen",
    )

    # Functions in the same module share __globals__, so track per-dict
    globals_id = id(func.__globals__)
    if globals_id not in _globals_state:
        originals: dict[str, Any] = {}
        for name in dispatch_names:
            if name in func.__globals__:
                originals[name] = func.__globals__[name]
        _globals_state[globals_id] = _GlobalsState(refcount=0, originals=originals)

    _globals_state[globals_id].refcount += 1

    new_kwdefaults: dict[str, Any] = dict(template.__kwdefaults__ or {})
    new_kwdefaults["__tenro_id__"] = func_id

    try:
        func.__globals__["_tenro_dispatch_sync"] = _tenro_dispatch_sync
        func.__globals__["_tenro_dispatch_async"] = _tenro_dispatch_async
        func.__globals__["_tenro_dispatch_gen"] = _tenro_dispatch_gen
        func.__globals__["_tenro_dispatch_asyncgen"] = _tenro_dispatch_asyncgen

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*code.*")
            func.__code__ = template.__code__
            func.__kwdefaults__ = new_kwdefaults

        _REGISTRY[func_id] = _RegisteredTarget(
            original_code=original_code,
            original_defaults=original_defaults,
            original_kwdefaults=original_kwdefaults,
            func_globals=func.__globals__,
            canonical_key=canonical_key,
            kind=kind,
        )
        _patched[id(func)] = func_id

    except BaseException:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*code.*")
            func.__code__ = original_code
            func.__defaults__ = original_defaults
            func.__kwdefaults__ = original_kwdefaults

        _globals_state[globals_id].refcount -= 1
        if _globals_state[globals_id].refcount == 0:
            state = _globals_state.pop(globals_id)
            for name in dispatch_names:
                if name in state.originals:
                    func.__globals__[name] = state.originals[name]
                else:
                    func.__globals__.pop(name, None)

        raise


def uninstall_trampoline(func: Callable[..., Any]) -> None:
    """Restore the original __code__ for a patched function.

    Args:
        func: The function to restore.
    """
    func_id = _patched.pop(id(func), None)
    if func_id is None:
        return

    reg = _REGISTRY.pop(func_id, None)
    if reg is None:
        return

    if inspect.isfunction(func):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*code.*")
            func.__code__ = reg.original_code
            func.__defaults__ = reg.original_defaults
            func.__kwdefaults__ = reg.original_kwdefaults

        globals_id = id(func.__globals__)
        state = _globals_state.get(globals_id)
        if state:
            state.refcount -= 1
            if state.refcount == 0:
                dispatch_names = (
                    "_tenro_dispatch_sync",
                    "_tenro_dispatch_async",
                    "_tenro_dispatch_gen",
                    "_tenro_dispatch_asyncgen",
                )
                for name in dispatch_names:
                    if name in state.originals:
                        func.__globals__[name] = state.originals[name]
                    else:
                        func.__globals__.pop(name, None)
                del _globals_state[globals_id]


def uninstall_all_trampolines() -> None:
    """Restore all patched functions to their original state.

    The orchestrator tracks function objects via _registered_trampolines
    and calls uninstall_trampoline() on each. This function clears
    the registry for any orphaned entries.
    """
    _patched.clear()
    _REGISTRY.clear()
    _globals_state.clear()
