# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Decorator-based linking for agents, LLMs, and tools.

Context-aware decorators that register functions with the Tenro system:
- Active Construct: Creates spans for testing and verification.
- No Construct: Executes the function without span tracking.

Features:
- Re-entrancy guard: Method delegation creates only one span.
- Multi-method wrapping: Classes get ALL matching entry methods wrapped.
- Framework object support: Patches invoke/run on pre-constructed objects.
- Agent attribution: HTTP interceptor uses span stack for agent correlation.

Examples:
    >>> from tenro import Construct, link_agent
    >>>
    >>> @link_agent("Manager")
    ... def manager_agent(task: str) -> str:
    ...     value = worker_agent(task)
    ...     pass
    >>>
    >>> @link_agent("WriterAgent")
    ... class WriterAgent:
    ...     async def execute(self, prompt: str) -> str:
    ...         pass
    >>>
    >>> with Construct() as construct:
    ...     data = manager_agent("Build feature")
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast, overload
from weakref import WeakKeyDictionary

from uuid_utils import uuid7

from tenro._core.spans import AgentRun, LLMCall, LLMScope, ToolCall
from tenro.errors import TenroError, TenroTracingWarning, warn
from tenro.linking.constants import (
    AGENT_ENTRY_METHODS,
    AGENT_ENTRY_PRECEDENCE,
    TOOL_ENTRY_METHODS,
    TOOL_ENTRY_PRECEDENCE,
)
from tenro.linking.context import (
    GuardKey,
    get_active_construct,
    guard_enter,
    guard_exit,
    set_active_construct,
)
from tenro.linking.detection import TargetType, detect_target_type
from tenro.linking.dispatch import (
    dispatch_async,
    dispatch_asyncgen,
    dispatch_gen,
    dispatch_sync,
    get_simulation_model,
)
from tenro.linking.generators import wrap_async_generator, wrap_generator
from tenro.linking.metadata import (
    ATTR_DISPLAY_NAME,
    ATTR_LINKED_METHODS,
    ATTR_LINKED_TYPE,
    ATTR_TARGET_PATHS,
    ATTR_WRAPPED,
    find_entry_methods,
    is_directly_linked,
)
from tenro.util import format_file_location
from tenro.util.env import get_env_bool

F = TypeVar("F", bound=Callable[..., Any])

_get_active_construct = get_active_construct
_set_active_construct = set_active_construct

_identity_fallback: WeakKeyDictionary[object, str] = WeakKeyDictionary()
ATTR_IDENTITY_KEY = "_tenro_identity_key"


def _stamp_identity_on_original(func: Callable[..., Any], target_path: str) -> None:
    """Stamp identity on the original callable for bidirectional resolution.

    Args:
        func: The original callable (may be wrapped by other decorators).
        target_path: The canonical dotted path for this target.
    """
    original = inspect.unwrap(func)
    try:
        setattr(original, ATTR_IDENTITY_KEY, target_path)
    except (AttributeError, TypeError):
        _identity_fallback[original] = target_path


def get_identity_from_original(func: Callable[..., Any]) -> str | None:
    """Get identity from an original callable.

    Args:
        func: The callable to check (may be original or wrapper).

    Returns:
        The canonical target path if found, None otherwise.
    """
    original = inspect.unwrap(func)
    identity: str | None = getattr(original, ATTR_IDENTITY_KEY, None)
    if identity is not None:
        return identity
    return _identity_fallback.get(original)


def _is_linking_enabled() -> bool:
    """Check if decorator linking is enabled via environment."""
    return get_env_bool("TENRO_LINKING_ENABLED", default=True)


def _resolve_entry_points(
    explicit_entry_points: str | list[str] | None,
    target: type | object,
    decorator_type: str,
    display_name: str,
    default_precedence: tuple[str, ...],
) -> list[str]:
    """Resolve and validate entry_points for class/object decoration.

    Args:
        explicit_entry_points: User-provided entry_points (str, list, or None).
        target: The class or object being decorated.
        decorator_type: "agent" or "tool" (for error messages).
        display_name: The display name for error messages.
        default_precedence: Default entry method precedence if not specified.

    Returns:
        List of method names to wrap.

    Raises:
        ValueError: If specified methods don't exist on target.
    """
    if explicit_entry_points is not None:
        entry_list = (
            [explicit_entry_points]
            if isinstance(explicit_entry_points, str)
            else list(explicit_entry_points)
        )
        missing = [m for m in entry_list if not hasattr(target, m)]
        if missing:
            target_kind = "class" if isinstance(target, type) else "object"
            raise ValueError(
                f"@link_{decorator_type}('{display_name}', "
                f"entry_points={explicit_entry_points!r}): "
                f"{target_kind} has no method(s): {', '.join(missing)}"
            )
        return entry_list
    else:
        return list(default_precedence)


def _check_class_not_linked(cls: type, new_type: str) -> None:
    """Guard against double-linking or cross-linking a class.

    Args:
        cls: The class being decorated.
        new_type: The link type being applied ("agent", "tool", "llm").

    Raises:
        TenroError: If class is already linked.
    """
    if is_directly_linked(cls):
        existing_type = getattr(cls, ATTR_LINKED_TYPE, "unknown")
        if existing_type == new_type:
            raise TenroError(
                f"Class '{cls.__name__}' is already decorated with @link_{new_type}. "
                f"Remove the duplicate decorator."
            )
        else:
            raise TenroError(
                f"Class '{cls.__name__}' is already decorated with @link_{existing_type}. "
                f"A class cannot be both a {existing_type} and a {new_type}."
            )


def _get_descriptor_from_mro(cls: type, method_name: str) -> Any:
    """Get raw descriptor from class MRO, checking inherited classes.

    This is needed to preserve staticmethod/classmethod wrappers for inherited methods.
    Using cls.__dict__.get() only finds directly-defined attributes, not inherited ones.

    Args:
        cls: Class to search from.
        method_name: Method name to find.

    Returns:
        Raw descriptor (staticmethod/classmethod/function) or None if not found.
    """
    for klass in cls.__mro__:
        if method_name in klass.__dict__:
            return klass.__dict__[method_name]
    return None


def _wrap_agent_method(
    method: Callable[..., Any],
    agent_name: str,
    method_target_path: str | None = None,
) -> Callable[..., Any]:
    """Wrap a method with agent span tracking and re-entrancy guard.

    Args:
        method: The method to wrap.
        agent_name: Display name for the agent span.
        method_target_path: Fully qualified path for span identity (e.g., "mymod.Cls.run").
    """
    if getattr(method, ATTR_WRAPPED, False):
        return method

    if inspect.isasyncgenfunction(method):

        @wraps(method)
        async def asyncgen_wrapper(
            self: Any, *args: Any, **kwargs: Any
        ) -> Any:  # Returns AsyncGenerator
            key = GuardKey(kind="agent", target_id=id(self))
            token = guard_enter(key)
            if token is None:
                async for item in method(self, *args, **kwargs):
                    yield item
                return

            construct = get_active_construct()
            if not construct:
                try:
                    async for item in method(self, *args, **kwargs):
                        yield item
                finally:
                    guard_exit(token)
                return

            canonical_key = method_target_path or f"unknown.{agent_name}"
            span = AgentRun(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=canonical_key,
                display_name=agent_name,
                input_data=args,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            error: Exception | None = None
            try:
                gen, simulated = await dispatch_asyncgen(
                    canonical_key, method, (self, *args), kwargs
                )
                span.simulated = simulated
                async for item in gen:
                    yield item
            except Exception as e:
                error = e
                raise
            finally:
                try:
                    if error is not None:
                        lifecycle.error_span_manual(span, parent_span_id, error)
                    else:
                        lifecycle.end_span_manual(span, parent_span_id)
                finally:
                    guard_exit(token)

        setattr(asyncgen_wrapper, ATTR_WRAPPED, True)
        if method_target_path:
            setattr(asyncgen_wrapper, ATTR_TARGET_PATHS, (method_target_path,))
            setattr(asyncgen_wrapper, ATTR_DISPLAY_NAME, agent_name)
        return asyncgen_wrapper

    if inspect.iscoroutinefunction(method):

        @wraps(method)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="agent", target_id=id(self))
            token = guard_enter(key)
            if token is None:
                return await method(self, *args, **kwargs)  # Re-entry: pass-through

            construct = get_active_construct()
            if not construct:
                try:
                    return await method(self, *args, **kwargs)
                finally:
                    guard_exit(token)

            canonical_key = method_target_path or f"unknown.{agent_name}"
            span = AgentRun(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=canonical_key,
                display_name=agent_name,
                input_data=args,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            try:
                result, simulated = await dispatch_async(
                    canonical_key, method, (self, *args), kwargs
                )
            except Exception as e:
                lifecycle.error_span_manual(span, parent_span_id, e)
                guard_exit(token)
                raise

            span.simulated = simulated
            if inspect.isasyncgen(result):
                return wrap_async_generator(
                    result, span, parent_span_id, lifecycle, token, guard_exit
                )

            span.output_data = result
            lifecycle.end_span_manual(span, parent_span_id)
            guard_exit(token)
            return result

        setattr(async_wrapper, ATTR_WRAPPED, True)
        if method_target_path:
            setattr(async_wrapper, ATTR_TARGET_PATHS, (method_target_path,))
            setattr(async_wrapper, ATTR_DISPLAY_NAME, agent_name)
        return async_wrapper

    if inspect.isgeneratorfunction(method):

        @wraps(method)
        def gen_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="agent", target_id=id(self))
            token = guard_enter(key)
            if token is None:
                return method(self, *args, **kwargs)

            construct = get_active_construct()
            if not construct:
                try:
                    return method(self, *args, **kwargs)
                finally:
                    guard_exit(token)

            canonical_key = method_target_path or f"unknown.{agent_name}"
            span = AgentRun(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=canonical_key,
                display_name=agent_name,
                input_data=args,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            try:
                gen, simulated = dispatch_gen(canonical_key, method, (self, *args), kwargs)
            except Exception as e:
                lifecycle.error_span_manual(span, parent_span_id, e)
                guard_exit(token)
                raise

            span.simulated = simulated
            return wrap_generator(gen, span, parent_span_id, lifecycle, token, guard_exit)

        setattr(gen_wrapper, ATTR_WRAPPED, True)
        if method_target_path:
            setattr(gen_wrapper, ATTR_TARGET_PATHS, (method_target_path,))
            setattr(gen_wrapper, ATTR_DISPLAY_NAME, agent_name)
        return gen_wrapper

    @wraps(method)
    def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        key = GuardKey(kind="agent", target_id=id(self))
        token = guard_enter(key)
        if token is None:
            return method(self, *args, **kwargs)  # Re-entry: pass-through

        construct = get_active_construct()
        if not construct:
            try:
                return method(self, *args, **kwargs)
            finally:
                guard_exit(token)

        canonical_key = method_target_path or f"unknown.{agent_name}"
        span = AgentRun(
            id=str(uuid7()),
            trace_id=str(uuid7()),
            start_time=time.time(),
            target_path=canonical_key,
            display_name=agent_name,
            input_data=args,
            kwargs=kwargs,
        )
        lifecycle = construct._lifecycle
        parent_span_id = lifecycle.start_span_manual(span)

        try:
            result, simulated = dispatch_sync(canonical_key, method, (self, *args), kwargs)
        except Exception as e:
            lifecycle.error_span_manual(span, parent_span_id, e)
            guard_exit(token)
            raise

        span.simulated = simulated
        if inspect.isgenerator(result):
            return wrap_generator(result, span, parent_span_id, lifecycle, token, guard_exit)

        span.output_data = result
        lifecycle.end_span_manual(span, parent_span_id)
        guard_exit(token)
        return result

    setattr(sync_wrapper, ATTR_WRAPPED, True)
    if method_target_path:
        setattr(sync_wrapper, ATTR_TARGET_PATHS, (method_target_path,))
        setattr(sync_wrapper, ATTR_DISPLAY_NAME, agent_name)
    return sync_wrapper


def _wrap_agent_function(func: F, agent_name: str) -> F:
    """Wrap a function with agent span tracking and re-entrancy guard."""
    if getattr(func, ATTR_WRAPPED, False):
        return func

    target_path = f"{func.__module__}.{func.__qualname__}"

    if inspect.isasyncgenfunction(func):

        @wraps(func)
        async def asyncgen_wrapper(*args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="agent", target_id=id(func))
            token = guard_enter(key)
            if token is None:
                async for item in func(*args, **kwargs):
                    yield item
                return

            construct = get_active_construct()
            if not construct:
                try:
                    async for item in func(*args, **kwargs):
                        yield item
                finally:
                    guard_exit(token)
                return

            span = AgentRun(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=target_path,
                display_name=agent_name,
                input_data=args,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            error: Exception | None = None
            try:
                gen, simulated = await dispatch_asyncgen(target_path, func, args, kwargs)
                span.simulated = simulated
                async for item in gen:
                    yield item
            except Exception as e:
                error = e
                raise
            finally:
                try:
                    if error is not None:
                        lifecycle.error_span_manual(span, parent_span_id, error)
                    else:
                        lifecycle.end_span_manual(span, parent_span_id)
                finally:
                    guard_exit(token)

        setattr(asyncgen_wrapper, ATTR_WRAPPED, True)
        setattr(asyncgen_wrapper, ATTR_LINKED_TYPE, "agent")
        setattr(asyncgen_wrapper, ATTR_TARGET_PATHS, (target_path,))
        setattr(asyncgen_wrapper, ATTR_DISPLAY_NAME, agent_name)
        asyncgen_wrapper._tenro_linked_name = agent_name  # type: ignore[attr-defined]
        asyncgen_wrapper._tenro_full_path = f"{func.__module__}.{func.__name__}"  # type: ignore[attr-defined]
        _stamp_identity_on_original(func, target_path)
        return asyncgen_wrapper  # type: ignore[return-value]

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="agent", target_id=id(func))
            token = guard_enter(key)
            if token is None:
                return await func(*args, **kwargs)  # Re-entry: pass-through

            construct = get_active_construct()
            if not construct:
                try:
                    return await func(*args, **kwargs)
                finally:
                    guard_exit(token)

            span = AgentRun(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=target_path,
                display_name=agent_name,
                input_data=args,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            try:
                result, simulated = await dispatch_async(target_path, func, args, kwargs)
            except Exception as e:
                lifecycle.error_span_manual(span, parent_span_id, e)
                guard_exit(token)
                raise

            span.simulated = simulated
            if inspect.isasyncgen(result):
                return wrap_async_generator(
                    result, span, parent_span_id, lifecycle, token, guard_exit
                )

            span.output_data = result
            lifecycle.end_span_manual(span, parent_span_id)
            guard_exit(token)
            return result

        setattr(async_wrapper, ATTR_WRAPPED, True)
        setattr(async_wrapper, ATTR_LINKED_TYPE, "agent")
        setattr(async_wrapper, ATTR_TARGET_PATHS, (target_path,))
        setattr(async_wrapper, ATTR_DISPLAY_NAME, agent_name)
        async_wrapper._tenro_linked_name = agent_name  # type: ignore[attr-defined]
        async_wrapper._tenro_full_path = f"{func.__module__}.{func.__name__}"  # type: ignore[attr-defined]
        _stamp_identity_on_original(func, target_path)
        return async_wrapper  # type: ignore[return-value]

    if inspect.isgeneratorfunction(func):

        @wraps(func)
        def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="agent", target_id=id(func))
            token = guard_enter(key)
            if token is None:
                return func(*args, **kwargs)

            construct = get_active_construct()
            if not construct:
                try:
                    return func(*args, **kwargs)
                finally:
                    guard_exit(token)

            span = AgentRun(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=target_path,
                display_name=agent_name,
                input_data=args,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            try:
                gen, simulated = dispatch_gen(target_path, func, args, kwargs)
            except Exception as e:
                lifecycle.error_span_manual(span, parent_span_id, e)
                guard_exit(token)
                raise

            span.simulated = simulated
            return wrap_generator(gen, span, parent_span_id, lifecycle, token, guard_exit)

        setattr(gen_wrapper, ATTR_WRAPPED, True)
        setattr(gen_wrapper, ATTR_LINKED_TYPE, "agent")
        setattr(gen_wrapper, ATTR_TARGET_PATHS, (target_path,))
        setattr(gen_wrapper, ATTR_DISPLAY_NAME, agent_name)
        gen_wrapper._tenro_linked_name = agent_name  # type: ignore[attr-defined]
        gen_wrapper._tenro_full_path = f"{func.__module__}.{func.__name__}"  # type: ignore[attr-defined]
        _stamp_identity_on_original(func, target_path)
        return gen_wrapper  # type: ignore[return-value]

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        key = GuardKey(kind="agent", target_id=id(func))
        token = guard_enter(key)
        if token is None:
            return func(*args, **kwargs)  # Re-entry: pass-through

        construct = get_active_construct()
        if not construct:
            try:
                return func(*args, **kwargs)
            finally:
                guard_exit(token)

        span = AgentRun(
            id=str(uuid7()),
            trace_id=str(uuid7()),
            start_time=time.time(),
            target_path=target_path,
            display_name=agent_name,
            input_data=args,
        )
        lifecycle = construct._lifecycle
        parent_span_id = lifecycle.start_span_manual(span)

        try:
            result, simulated = dispatch_sync(target_path, func, args, kwargs)
        except Exception as e:
            lifecycle.error_span_manual(span, parent_span_id, e)
            guard_exit(token)
            raise

        span.output_data = result
        span.simulated = simulated
        lifecycle.end_span_manual(span, parent_span_id)
        guard_exit(token)
        return result

    setattr(sync_wrapper, ATTR_WRAPPED, True)
    setattr(sync_wrapper, ATTR_LINKED_TYPE, "agent")
    setattr(sync_wrapper, ATTR_TARGET_PATHS, (target_path,))
    setattr(sync_wrapper, ATTR_DISPLAY_NAME, agent_name)
    sync_wrapper._tenro_linked_name = agent_name  # type: ignore[attr-defined]
    sync_wrapper._tenro_full_path = f"{func.__module__}.{func.__name__}"  # type: ignore[attr-defined]
    _stamp_identity_on_original(func, target_path)
    return sync_wrapper  # type: ignore[return-value]


def _decorate_agent_class(
    cls: type, agent_name: str, explicit_entry_points: str | list[str] | None
) -> type:
    """Decorate a class by wrapping all matching entry methods."""
    _check_class_not_linked(cls, "agent")

    if explicit_entry_points is not None:
        entry_list = (
            [explicit_entry_points]
            if isinstance(explicit_entry_points, str)
            else explicit_entry_points
        )
        missing = [m for m in entry_list if not hasattr(cls, m)]
        if missing:
            raise ValueError(
                f"@link_agent('{agent_name}', entry_points={explicit_entry_points!r}): "
                f"class has no method(s): {', '.join(missing)}"
            )
        methods_to_wrap = entry_list
    else:
        methods_to_wrap = find_entry_methods(cls, AGENT_ENTRY_PRECEDENCE)

    if not methods_to_wrap:
        raise ValueError(
            f"@link_agent('{cls.__name__}'): could not find entry method.\n"
            f"Expected one of: {', '.join(sorted(AGENT_ENTRY_METHODS))}\n"
            f"Either add one of these methods or specify explicitly:\n"
            f"  @link_agent('{cls.__name__}', entry_points='your_method')"
        )

    wrapped_methods: list[str] = []
    all_target_paths: list[str] = []
    for method_name in methods_to_wrap:
        method_target_path = f"{cls.__module__}.{cls.__qualname__}.{method_name}"

        # Check for descriptors in class MRO (handles inherited staticmethod/classmethod)
        # This preserves staticmethod/classmethod wrappers for both direct and inherited methods
        raw = _get_descriptor_from_mro(cls, method_name)
        if isinstance(raw, staticmethod):
            wrapped_fn = _wrap_agent_method(raw.__func__, agent_name, method_target_path)
            to_set: staticmethod[..., Any] | classmethod[Any, ..., Any] | Callable[..., Any]
            to_set = staticmethod(wrapped_fn)
        elif isinstance(raw, classmethod):
            wrapped_fn = _wrap_agent_method(raw.__func__, agent_name, method_target_path)
            to_set = classmethod(wrapped_fn)
        else:
            original = getattr(cls, method_name)
            to_set = _wrap_agent_method(original, agent_name, method_target_path)

        try:
            setattr(cls, method_name, to_set)
            wrapped_methods.append(method_name)
            all_target_paths.append(method_target_path)
        except (TypeError, AttributeError) as e:
            warn(
                f"Tenro: Cannot wrap {cls.__name__}.{method_name}: {e}. "
                "Extended tracing unavailable.",
                TenroTracingWarning,
                stacklevel=3,
            )

    setattr(cls, ATTR_LINKED_TYPE, "agent")
    setattr(cls, ATTR_DISPLAY_NAME, agent_name)
    setattr(cls, ATTR_TARGET_PATHS, tuple(all_target_paths))
    setattr(cls, ATTR_LINKED_METHODS, tuple(wrapped_methods))
    cls._tenro_linked_name = agent_name  # type: ignore[attr-defined]
    return cls


def _patch_agent_object(
    obj: object, agent_name: str, explicit_entry_points: str | list[str] | None
) -> object:
    """Patch entry methods on a framework object instance.

    For objects with only __call__ as entry method, returns a proxy wrapper
    since Python looks up dunder methods on the class, not the instance.
    """
    patched_any = False
    patched_target_paths: list[str] = []
    obj_type = type(obj)
    call_only = False

    if explicit_entry_points is not None:
        entry_list = (
            [explicit_entry_points]
            if isinstance(explicit_entry_points, str)
            else explicit_entry_points
        )
        missing = [m for m in entry_list if not hasattr(obj, m) or not callable(getattr(obj, m))]
        if missing:
            raise ValueError(
                f"@link_agent('{agent_name}', entry_points={explicit_entry_points!r}): "
                f"object has no method(s): {', '.join(missing)}"
            )
        methods_to_patch = entry_list
    else:
        methods_to_patch = [m for m in AGENT_ENTRY_PRECEDENCE if m != "__call__"]

    for method_name in methods_to_patch:
        if method_name == "__call__":
            continue
        original = getattr(obj, method_name, None)
        if original is not None and callable(original):
            method_target_path = f"{obj_type.__module__}.{obj_type.__qualname__}.{method_name}"
            wrapped = _make_agent_object_wrapper(original, agent_name, obj, method_target_path)
            try:
                setattr(obj, method_name, wrapped)
                patched_any = True
                patched_target_paths.append(method_target_path)
            except (TypeError, AttributeError) as e:
                warn(
                    f"Tenro: Cannot patch {obj_type.__name__}.{method_name}: {e}. "
                    "Extended tracing unavailable.",
                    TenroTracingWarning,
                    stacklevel=3,
                )

    call_requested = explicit_entry_points is not None and "__call__" in (
        [explicit_entry_points] if isinstance(explicit_entry_points, str) else explicit_entry_points
    )
    should_check_call = call_requested or (not patched_any and explicit_entry_points is None)
    if should_check_call and "__call__" in obj_type.__dict__:
        call_only = True
        method_target_path = f"{obj_type.__module__}.{obj_type.__qualname__}.__call__"
        patched_target_paths.append(method_target_path)

    if not patched_any and not call_only:
        raise ValueError(
            f"@link_agent('{agent_name}'): could not find entry method on {obj_type.__name__}"
        )

    if call_only:
        return _make_callable_proxy(
            obj, agent_name, patched_target_paths[0], _make_agent_object_wrapper, "agent"
        )

    try:
        setattr(obj, ATTR_TARGET_PATHS, tuple(patched_target_paths))
        setattr(obj, ATTR_DISPLAY_NAME, agent_name)
        setattr(obj, ATTR_LINKED_TYPE, "agent")
    except (TypeError, AttributeError) as e:
        # Frozen objects (e.g., dataclasses with frozen=True) cannot be mutated
        warn(
            f"Tenro: Cannot store metadata on {obj_type.__name__}: {e}. "
            "Object target verification may be limited.",
            TenroTracingWarning,
            stacklevel=3,
        )

    return obj


def _make_callable_proxy(
    wrapped_obj: object,
    display_name: str,
    target_path: str,
    wrapper_maker: Callable[..., Callable[..., Any]],
    linked_type: str,
) -> object:
    """Create a proxy for callable objects with only __call__.

    Python looks up dunder methods on the class, not instance __dict__.
    This proxy class has a wrapped __call__ that creates spans.

    Args:
        wrapped_obj: The original callable object to wrap.
        display_name: Display name for spans (agent/tool name).
        target_path: Fully qualified path for span identity.
        wrapper_maker: Function to create the wrapper (agent or tool).
        linked_type: Type of linking ("agent" or "tool").
    """
    original_call = wrapped_obj.__call__  # type: ignore[operator]
    obj_type = type(wrapped_obj)

    call_wrapper = wrapper_maker(original_call, display_name, wrapped_obj, target_path)

    class CallableProxy(obj_type):  # type: ignore[valid-type,misc]
        """Proxy wrapper for callable objects."""

        _tenro_wrapped_obj: object

        def __new__(cls) -> CallableProxy:
            return object.__new__(cls)

        def __init__(self) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return call_wrapper(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._tenro_wrapped_obj, name)

        def __setattr__(self, name: str, value: Any) -> None:
            if name == "_tenro_wrapped_obj":
                object.__setattr__(self, name, value)
            else:
                setattr(self._tenro_wrapped_obj, name, value)

    proxy = CallableProxy()
    proxy._tenro_wrapped_obj = wrapped_obj
    setattr(proxy, ATTR_TARGET_PATHS, (target_path,))
    setattr(proxy, ATTR_DISPLAY_NAME, display_name)
    setattr(proxy, ATTR_LINKED_TYPE, linked_type)

    return proxy


def _make_agent_object_wrapper(
    original: Callable[..., Any],
    agent_name: str,
    obj: object,
    method_target_path: str | None = None,
) -> Callable[..., Any]:
    """Create wrapper for framework object method.

    Args:
        original: The original method to wrap.
        agent_name: Display name for the agent span.
        obj: The object instance being patched (for re-entrancy guard).
        method_target_path: Fully qualified path for span identity.
    """
    if getattr(original, ATTR_WRAPPED, False):
        return original

    if inspect.isasyncgenfunction(original):

        @wraps(original)
        async def asyncgen_wrapper(*args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="agent", target_id=id(obj))
            token = guard_enter(key)
            if token is None:
                async for item in original(*args, **kwargs):
                    yield item
                return

            construct = get_active_construct()
            if not construct:
                try:
                    async for item in original(*args, **kwargs):
                        yield item
                finally:
                    guard_exit(token)
                return

            span = AgentRun(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=method_target_path or f"unknown.{agent_name}",
                display_name=agent_name,
                input_data=args,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            canonical_key = method_target_path or f"unknown.{agent_name}"
            error: Exception | None = None
            try:
                gen, simulated = await dispatch_asyncgen(canonical_key, original, args, kwargs)
                span.simulated = simulated
                async for item in gen:
                    yield item
            except Exception as e:
                error = e
                raise
            finally:
                try:
                    if error is not None:
                        lifecycle.error_span_manual(span, parent_span_id, error)
                    else:
                        lifecycle.end_span_manual(span, parent_span_id)
                finally:
                    guard_exit(token)

        setattr(asyncgen_wrapper, ATTR_WRAPPED, True)
        if method_target_path:
            setattr(asyncgen_wrapper, ATTR_TARGET_PATHS, (method_target_path,))
            setattr(asyncgen_wrapper, ATTR_DISPLAY_NAME, agent_name)
        return asyncgen_wrapper

    if inspect.iscoroutinefunction(original):

        @wraps(original)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="agent", target_id=id(obj))
            token = guard_enter(key)
            if token is None:
                return await original(*args, **kwargs)

            construct = get_active_construct()
            if not construct:
                try:
                    return await original(*args, **kwargs)
                finally:
                    guard_exit(token)

            span = AgentRun(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=method_target_path or f"unknown.{agent_name}",
                display_name=agent_name,
                input_data=args,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            canonical_key = method_target_path or f"unknown.{agent_name}"
            try:
                result, simulated = await dispatch_async(canonical_key, original, args, kwargs)
            except Exception as e:
                lifecycle.error_span_manual(span, parent_span_id, e)
                guard_exit(token)
                raise

            span.simulated = simulated
            if inspect.isasyncgen(result):
                return wrap_async_generator(
                    result, span, parent_span_id, lifecycle, token, guard_exit
                )

            span.output_data = result
            lifecycle.end_span_manual(span, parent_span_id)
            guard_exit(token)
            return result

        setattr(async_wrapper, ATTR_WRAPPED, True)
        if method_target_path:
            setattr(async_wrapper, ATTR_TARGET_PATHS, (method_target_path,))
            setattr(async_wrapper, ATTR_DISPLAY_NAME, agent_name)
        return async_wrapper
    else:

        @wraps(original)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="agent", target_id=id(obj))
            token = guard_enter(key)
            if token is None:
                return original(*args, **kwargs)

            construct = get_active_construct()
            if not construct:
                try:
                    return original(*args, **kwargs)
                finally:
                    guard_exit(token)

            span = AgentRun(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=method_target_path or f"unknown.{agent_name}",
                display_name=agent_name,
                input_data=args,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            canonical_key = method_target_path or f"unknown.{agent_name}"
            try:
                result, simulated = dispatch_sync(canonical_key, original, args, kwargs)
            except Exception as e:
                lifecycle.error_span_manual(span, parent_span_id, e)
                guard_exit(token)
                raise

            span.simulated = simulated
            if inspect.isgenerator(result):
                return wrap_generator(result, span, parent_span_id, lifecycle, token, guard_exit)

            span.output_data = result
            lifecycle.end_span_manual(span, parent_span_id)
            guard_exit(token)
            return result

        setattr(sync_wrapper, ATTR_WRAPPED, True)
        if method_target_path:
            setattr(sync_wrapper, ATTR_TARGET_PATHS, (method_target_path,))
            setattr(sync_wrapper, ATTR_DISPLAY_NAME, agent_name)
        return sync_wrapper


def _wrap_tool_function(func: F, tool_name: str) -> F:
    """Wrap a function with tool span tracking and re-entrancy guard."""
    if getattr(func, ATTR_WRAPPED, False):
        return func

    target_path = f"{func.__module__}.{func.__qualname__}"

    if inspect.isasyncgenfunction(func):
        # Async generator function
        @wraps(func)
        async def asyncgen_wrapper(*args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="tool", target_id=id(func))
            token = guard_enter(key)
            if token is None:
                async for item in func(*args, **kwargs):
                    yield item
                return

            construct = get_active_construct()
            if not construct:
                try:
                    async for item in func(*args, **kwargs):
                        yield item
                finally:
                    guard_exit(token)
                return

            span = ToolCall(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=target_path,
                display_name=tool_name,
                args=args,
                kwargs=kwargs,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            error: Exception | None = None
            try:
                gen, simulated = await dispatch_asyncgen(target_path, func, args, kwargs)
                span.simulated = simulated
                async for item in gen:
                    yield item
            except Exception as e:
                error = e
                raise
            finally:
                try:
                    if error is not None:
                        lifecycle.error_span_manual(span, parent_span_id, error)
                    else:
                        lifecycle.end_span_manual(span, parent_span_id)
                finally:
                    guard_exit(token)

        setattr(asyncgen_wrapper, ATTR_WRAPPED, True)
        setattr(asyncgen_wrapper, ATTR_LINKED_TYPE, "tool")
        setattr(asyncgen_wrapper, ATTR_TARGET_PATHS, (target_path,))
        setattr(asyncgen_wrapper, ATTR_DISPLAY_NAME, tool_name)
        asyncgen_wrapper._tenro_linked_name = tool_name  # type: ignore[attr-defined]
        asyncgen_wrapper._tenro_full_path = f"{func.__module__}.{func.__name__}"  # type: ignore[attr-defined]
        _stamp_identity_on_original(func, target_path)
        return asyncgen_wrapper  # type: ignore[return-value]

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="tool", target_id=id(func))
            token = guard_enter(key)
            if token is None:
                return await func(*args, **kwargs)

            construct = get_active_construct()
            if not construct:
                try:
                    return await func(*args, **kwargs)
                finally:
                    guard_exit(token)

            span = ToolCall(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=target_path,
                display_name=tool_name,
                args=args,
                kwargs=kwargs,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            try:
                result, simulated = await dispatch_async(target_path, func, args, kwargs)
            except Exception as e:
                lifecycle.error_span_manual(span, parent_span_id, e)
                guard_exit(token)
                raise

            span.simulated = simulated
            if inspect.isasyncgen(result):
                return wrap_async_generator(
                    result, span, parent_span_id, lifecycle, token, guard_exit
                )

            span.result = result
            lifecycle.end_span_manual(span, parent_span_id)
            guard_exit(token)
            return result

        setattr(async_wrapper, ATTR_WRAPPED, True)
        setattr(async_wrapper, ATTR_LINKED_TYPE, "tool")
        setattr(async_wrapper, ATTR_TARGET_PATHS, (target_path,))
        setattr(async_wrapper, ATTR_DISPLAY_NAME, tool_name)
        async_wrapper._tenro_linked_name = tool_name  # type: ignore[attr-defined]
        async_wrapper._tenro_full_path = f"{func.__module__}.{func.__name__}"  # type: ignore[attr-defined]
        _stamp_identity_on_original(func, target_path)
        return async_wrapper  # type: ignore[return-value]

    if inspect.isgeneratorfunction(func):

        @wraps(func)
        def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="tool", target_id=id(func))
            token = guard_enter(key)
            if token is None:
                return func(*args, **kwargs)

            construct = get_active_construct()
            if not construct:
                try:
                    return func(*args, **kwargs)
                finally:
                    guard_exit(token)

            span = ToolCall(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=target_path,
                display_name=tool_name,
                args=args,
                kwargs=kwargs,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            try:
                gen, simulated = dispatch_gen(target_path, func, args, kwargs)
            except Exception as e:
                lifecycle.error_span_manual(span, parent_span_id, e)
                guard_exit(token)
                raise

            span.simulated = simulated
            return wrap_generator(gen, span, parent_span_id, lifecycle, token, guard_exit)

        setattr(gen_wrapper, ATTR_WRAPPED, True)
        setattr(gen_wrapper, ATTR_LINKED_TYPE, "tool")
        setattr(gen_wrapper, ATTR_TARGET_PATHS, (target_path,))
        setattr(gen_wrapper, ATTR_DISPLAY_NAME, tool_name)
        gen_wrapper._tenro_linked_name = tool_name  # type: ignore[attr-defined]
        gen_wrapper._tenro_full_path = f"{func.__module__}.{func.__name__}"  # type: ignore[attr-defined]
        _stamp_identity_on_original(func, target_path)
        return gen_wrapper  # type: ignore[return-value]

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        key = GuardKey(kind="tool", target_id=id(func))
        token = guard_enter(key)
        if token is None:
            return func(*args, **kwargs)

        construct = get_active_construct()
        if not construct:
            try:
                return func(*args, **kwargs)
            finally:
                guard_exit(token)

        span = ToolCall(
            id=str(uuid7()),
            trace_id=str(uuid7()),
            start_time=time.time(),
            target_path=target_path,
            display_name=tool_name,
            args=args,
            kwargs=kwargs,
        )
        lifecycle = construct._lifecycle
        parent_span_id = lifecycle.start_span_manual(span)

        try:
            result, simulated = dispatch_sync(target_path, func, args, kwargs)
        except Exception as e:
            lifecycle.error_span_manual(span, parent_span_id, e)
            guard_exit(token)
            raise

        span.result = result
        span.simulated = simulated
        lifecycle.end_span_manual(span, parent_span_id)
        guard_exit(token)
        return result

    setattr(sync_wrapper, ATTR_WRAPPED, True)
    setattr(sync_wrapper, ATTR_LINKED_TYPE, "tool")
    setattr(sync_wrapper, ATTR_TARGET_PATHS, (target_path,))
    setattr(sync_wrapper, ATTR_DISPLAY_NAME, tool_name)
    sync_wrapper._tenro_linked_name = tool_name  # type: ignore[attr-defined]
    sync_wrapper._tenro_full_path = f"{func.__module__}.{func.__name__}"  # type: ignore[attr-defined]
    _stamp_identity_on_original(func, target_path)
    return sync_wrapper  # type: ignore[return-value]


def _wrap_tool_method(
    method: Callable[..., Any],
    tool_name: str,
    method_target_path: str | None = None,
) -> Callable[..., Any]:
    """Wrap a method with tool span tracking and re-entrancy guard.

    Args:
        method: The method to wrap.
        tool_name: Display name for the tool span.
        method_target_path: Fully qualified path for span identity (e.g., "mymod.Cls.invoke").
    """
    if getattr(method, ATTR_WRAPPED, False):
        return method

    if inspect.isasyncgenfunction(method):
        # Async generator function
        @wraps(method)
        async def asyncgen_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="tool", target_id=id(self))
            token = guard_enter(key)
            if token is None:
                async for item in method(self, *args, **kwargs):
                    yield item
                return

            construct = get_active_construct()
            if not construct:
                try:
                    async for item in method(self, *args, **kwargs):
                        yield item
                finally:
                    guard_exit(token)
                return

            canonical_key = method_target_path or f"unknown.{tool_name}"
            span = ToolCall(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=canonical_key,
                display_name=tool_name,
                args=args,
                kwargs=kwargs,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            error: Exception | None = None
            try:
                gen, simulated = await dispatch_asyncgen(
                    canonical_key, method, (self, *args), kwargs
                )
                span.simulated = simulated
                async for item in gen:
                    yield item
            except Exception as e:
                error = e
                raise
            finally:
                try:
                    if error is not None:
                        lifecycle.error_span_manual(span, parent_span_id, error)
                    else:
                        lifecycle.end_span_manual(span, parent_span_id)
                finally:
                    guard_exit(token)

        setattr(asyncgen_wrapper, ATTR_WRAPPED, True)
        if method_target_path:
            setattr(asyncgen_wrapper, ATTR_TARGET_PATHS, (method_target_path,))
            setattr(asyncgen_wrapper, ATTR_DISPLAY_NAME, tool_name)
        return asyncgen_wrapper

    if inspect.iscoroutinefunction(method):

        @wraps(method)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="tool", target_id=id(self))
            token = guard_enter(key)
            if token is None:
                return await method(self, *args, **kwargs)

            construct = get_active_construct()
            if not construct:
                try:
                    return await method(self, *args, **kwargs)
                finally:
                    guard_exit(token)

            canonical_key = method_target_path or f"unknown.{tool_name}"
            span = ToolCall(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=canonical_key,
                display_name=tool_name,
                args=args,
                kwargs=kwargs,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            try:
                result, simulated = await dispatch_async(
                    canonical_key, method, (self, *args), kwargs
                )
            except Exception as e:
                lifecycle.error_span_manual(span, parent_span_id, e)
                guard_exit(token)
                raise

            span.simulated = simulated
            if inspect.isasyncgen(result):
                return wrap_async_generator(
                    result, span, parent_span_id, lifecycle, token, guard_exit
                )

            span.result = result
            lifecycle.end_span_manual(span, parent_span_id)
            guard_exit(token)
            return result

        setattr(async_wrapper, ATTR_WRAPPED, True)
        if method_target_path:
            setattr(async_wrapper, ATTR_TARGET_PATHS, (method_target_path,))
            setattr(async_wrapper, ATTR_DISPLAY_NAME, tool_name)
        return async_wrapper

    if inspect.isgeneratorfunction(method):
        # Sync generator function - must use dispatch_gen for simulation

        @wraps(method)
        def gen_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="tool", target_id=id(self))
            token = guard_enter(key)
            if token is None:
                return method(self, *args, **kwargs)

            construct = get_active_construct()
            if not construct:
                try:
                    return method(self, *args, **kwargs)
                finally:
                    guard_exit(token)

            canonical_key = method_target_path or f"unknown.{tool_name}"
            span = ToolCall(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=canonical_key,
                display_name=tool_name,
                args=args,
                kwargs=kwargs,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            try:
                gen, simulated = dispatch_gen(canonical_key, method, (self, *args), kwargs)
            except Exception as e:
                lifecycle.error_span_manual(span, parent_span_id, e)
                guard_exit(token)
                raise

            span.simulated = simulated
            return wrap_generator(gen, span, parent_span_id, lifecycle, token, guard_exit)

        setattr(gen_wrapper, ATTR_WRAPPED, True)
        if method_target_path:
            setattr(gen_wrapper, ATTR_TARGET_PATHS, (method_target_path,))
            setattr(gen_wrapper, ATTR_DISPLAY_NAME, tool_name)
        return gen_wrapper

    # Sync non-generator function
    @wraps(method)
    def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        key = GuardKey(kind="tool", target_id=id(self))
        token = guard_enter(key)
        if token is None:
            return method(self, *args, **kwargs)

        construct = get_active_construct()
        if not construct:
            try:
                return method(self, *args, **kwargs)
            finally:
                guard_exit(token)

        canonical_key = method_target_path or f"unknown.{tool_name}"
        span = ToolCall(
            id=str(uuid7()),
            trace_id=str(uuid7()),
            start_time=time.time(),
            target_path=canonical_key,
            display_name=tool_name,
            args=args,
            kwargs=kwargs,
        )
        lifecycle = construct._lifecycle
        parent_span_id = lifecycle.start_span_manual(span)

        try:
            result, simulated = dispatch_sync(canonical_key, method, (self, *args), kwargs)
        except Exception as e:
            lifecycle.error_span_manual(span, parent_span_id, e)
            guard_exit(token)
            raise

        span.simulated = simulated
        if inspect.isgenerator(result):
            return wrap_generator(result, span, parent_span_id, lifecycle, token, guard_exit)

        span.result = result
        lifecycle.end_span_manual(span, parent_span_id)
        guard_exit(token)
        return result

    setattr(sync_wrapper, ATTR_WRAPPED, True)
    if method_target_path:
        setattr(sync_wrapper, ATTR_TARGET_PATHS, (method_target_path,))
        setattr(sync_wrapper, ATTR_DISPLAY_NAME, tool_name)
    return sync_wrapper


def _decorate_tool_class(
    cls: type, tool_name: str, explicit_entry_points: str | list[str] | None
) -> type:
    """Decorate a class by wrapping all matching tool entry methods."""
    _check_class_not_linked(cls, "tool")

    if explicit_entry_points is not None:
        entry_list = (
            [explicit_entry_points]
            if isinstance(explicit_entry_points, str)
            else explicit_entry_points
        )
        missing = [m for m in entry_list if not hasattr(cls, m)]
        if missing:
            raise ValueError(
                f"@link_tool('{tool_name}', entry_points={explicit_entry_points!r}): "
                f"class has no method(s): {', '.join(missing)}"
            )
        methods_to_wrap = entry_list
    else:
        methods_to_wrap = find_entry_methods(cls, TOOL_ENTRY_PRECEDENCE)

    if not methods_to_wrap:
        raise ValueError(
            f"@link_tool('{tool_name}'): could not find entry method on "
            f"{cls.__name__}.\nExpected one of: {', '.join(sorted(TOOL_ENTRY_METHODS))}\n"
            f"Either add one of these methods or specify explicitly:\n"
            f"  @link_tool('{tool_name}', entry_points='your_method')"
        )

    wrapped_methods: list[str] = []
    all_target_paths: list[str] = []
    for method_name in methods_to_wrap:
        method_target_path = f"{cls.__module__}.{cls.__qualname__}.{method_name}"

        # Check for descriptors in class MRO (handles inherited staticmethod/classmethod)
        # This preserves staticmethod/classmethod wrappers for both direct and inherited methods
        raw = _get_descriptor_from_mro(cls, method_name)
        if isinstance(raw, staticmethod):
            wrapped_fn = _wrap_tool_method(raw.__func__, tool_name, method_target_path)
            to_set: staticmethod[..., Any] | classmethod[Any, ..., Any] | Callable[..., Any]
            to_set = staticmethod(wrapped_fn)
        elif isinstance(raw, classmethod):
            wrapped_fn = _wrap_tool_method(raw.__func__, tool_name, method_target_path)
            to_set = classmethod(wrapped_fn)
        else:
            original = getattr(cls, method_name)
            to_set = _wrap_tool_method(original, tool_name, method_target_path)

        try:
            setattr(cls, method_name, to_set)
            wrapped_methods.append(method_name)
            all_target_paths.append(method_target_path)
        except (TypeError, AttributeError) as e:
            warn(
                f"Tenro: Cannot wrap {cls.__name__}.{method_name}: {e}. "
                "Extended tracing unavailable.",
                TenroTracingWarning,
                stacklevel=3,
            )

    setattr(cls, ATTR_LINKED_TYPE, "tool")
    setattr(cls, ATTR_DISPLAY_NAME, tool_name)
    setattr(cls, ATTR_TARGET_PATHS, tuple(all_target_paths))
    setattr(cls, ATTR_LINKED_METHODS, tuple(wrapped_methods))
    cls._tenro_linked_name = tool_name  # type: ignore[attr-defined]
    return cls


def _patch_tool_object(
    obj: object, tool_name: str, explicit_entry_points: str | list[str] | None
) -> object:
    """Patch entry methods on a framework tool object instance.

    For objects with only __call__ as entry method, returns a proxy wrapper
    since Python looks up dunder methods on the class, not the instance.
    """
    patched_any = False
    patched_target_paths: list[str] = []
    obj_type = type(obj)
    call_only = False

    if explicit_entry_points is not None:
        entry_list = (
            [explicit_entry_points]
            if isinstance(explicit_entry_points, str)
            else explicit_entry_points
        )
        missing = [m for m in entry_list if not hasattr(obj, m) or not callable(getattr(obj, m))]
        if missing:
            raise ValueError(
                f"@link_tool('{tool_name}', entry_points={explicit_entry_points!r}): "
                f"object has no method(s): {', '.join(missing)}"
            )
        methods_to_patch = entry_list
    else:
        methods_to_patch = [m for m in TOOL_ENTRY_PRECEDENCE if m != "__call__"]

    for method_name in methods_to_patch:
        if method_name == "__call__":
            continue
        original = getattr(obj, method_name, None)
        if original is not None and callable(original):
            method_target_path = f"{obj_type.__module__}.{obj_type.__qualname__}.{method_name}"
            wrapped = _make_tool_object_wrapper(original, tool_name, obj, method_target_path)
            try:
                setattr(obj, method_name, wrapped)
                patched_any = True
                patched_target_paths.append(method_target_path)
            except (TypeError, AttributeError) as e:
                warn(
                    f"Tenro: Cannot patch {obj_type.__name__}.{method_name}: {e}. "
                    "Extended tracing unavailable.",
                    TenroTracingWarning,
                    stacklevel=3,
                )

    call_requested = explicit_entry_points is not None and "__call__" in (
        [explicit_entry_points] if isinstance(explicit_entry_points, str) else explicit_entry_points
    )
    should_check_call = call_requested or (not patched_any and explicit_entry_points is None)
    if should_check_call and "__call__" in obj_type.__dict__:
        call_only = True
        method_target_path = f"{obj_type.__module__}.{obj_type.__qualname__}.__call__"
        patched_target_paths.append(method_target_path)

    if not patched_any and not call_only:
        raise ValueError(
            f"@link_tool('{tool_name}'): could not find entry method on {obj_type.__name__}"
        )

    if call_only:
        return _make_callable_proxy(
            obj, tool_name, patched_target_paths[0], _make_tool_object_wrapper, "tool"
        )

    try:
        setattr(obj, ATTR_TARGET_PATHS, tuple(patched_target_paths))
        setattr(obj, ATTR_DISPLAY_NAME, tool_name)
        setattr(obj, ATTR_LINKED_TYPE, "tool")
    except (TypeError, AttributeError) as e:
        # Frozen objects (e.g., dataclasses with frozen=True) cannot be mutated
        warn(
            f"Tenro: Cannot store metadata on {obj_type.__name__}: {e}. "
            "Object target verification may be limited.",
            TenroTracingWarning,
            stacklevel=3,
        )

    return obj


def _make_tool_object_wrapper(
    original: Callable[..., Any],
    tool_name: str,
    obj: object,
    method_target_path: str | None = None,
) -> Callable[..., Any]:
    """Create wrapper for framework tool object method.

    Args:
        original: The original method to wrap.
        tool_name: Display name for the tool span.
        obj: The object instance being patched (for re-entrancy guard).
        method_target_path: Fully qualified path for span identity.
    """
    if getattr(original, ATTR_WRAPPED, False):
        return original

    if inspect.isasyncgenfunction(original):

        @wraps(original)
        async def asyncgen_wrapper(*args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="tool", target_id=id(obj))
            token = guard_enter(key)
            if token is None:
                async for item in original(*args, **kwargs):
                    yield item
                return

            construct = get_active_construct()
            if not construct:
                try:
                    async for item in original(*args, **kwargs):
                        yield item
                finally:
                    guard_exit(token)
                return

            span = ToolCall(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=method_target_path or f"unknown.{tool_name}",
                display_name=tool_name,
                args=args,
                kwargs=kwargs,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            canonical_key = method_target_path or f"unknown.{tool_name}"
            error: Exception | None = None
            try:
                gen, simulated = await dispatch_asyncgen(canonical_key, original, args, kwargs)
                span.simulated = simulated
                async for item in gen:
                    yield item
            except Exception as e:
                error = e
                raise
            finally:
                try:
                    if error is not None:
                        lifecycle.error_span_manual(span, parent_span_id, error)
                    else:
                        lifecycle.end_span_manual(span, parent_span_id)
                finally:
                    guard_exit(token)

        setattr(asyncgen_wrapper, ATTR_WRAPPED, True)
        if method_target_path:
            setattr(asyncgen_wrapper, ATTR_TARGET_PATHS, (method_target_path,))
            setattr(asyncgen_wrapper, ATTR_DISPLAY_NAME, tool_name)
        return asyncgen_wrapper

    if inspect.iscoroutinefunction(original):

        @wraps(original)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="tool", target_id=id(obj))
            token = guard_enter(key)
            if token is None:
                return await original(*args, **kwargs)

            construct = get_active_construct()
            if not construct:
                try:
                    return await original(*args, **kwargs)
                finally:
                    guard_exit(token)

            span = ToolCall(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=method_target_path or f"unknown.{tool_name}",
                display_name=tool_name,
                args=args,
                kwargs=kwargs,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            canonical_key = method_target_path or f"unknown.{tool_name}"
            try:
                result, simulated = await dispatch_async(canonical_key, original, args, kwargs)
            except Exception as e:
                lifecycle.error_span_manual(span, parent_span_id, e)
                guard_exit(token)
                raise

            span.simulated = simulated
            if inspect.isasyncgen(result):
                return wrap_async_generator(
                    result, span, parent_span_id, lifecycle, token, guard_exit
                )

            span.result = result
            lifecycle.end_span_manual(span, parent_span_id)
            guard_exit(token)
            return result

        setattr(async_wrapper, ATTR_WRAPPED, True)
        if method_target_path:
            setattr(async_wrapper, ATTR_TARGET_PATHS, (method_target_path,))
            setattr(async_wrapper, ATTR_DISPLAY_NAME, tool_name)
        return async_wrapper
    else:

        @wraps(original)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            key = GuardKey(kind="tool", target_id=id(obj))
            token = guard_enter(key)
            if token is None:
                return original(*args, **kwargs)

            construct = get_active_construct()
            if not construct:
                try:
                    return original(*args, **kwargs)
                finally:
                    guard_exit(token)

            span = ToolCall(
                id=str(uuid7()),
                trace_id=str(uuid7()),
                start_time=time.time(),
                target_path=method_target_path or f"unknown.{tool_name}",
                display_name=tool_name,
                args=args,
                kwargs=kwargs,
            )
            lifecycle = construct._lifecycle
            parent_span_id = lifecycle.start_span_manual(span)

            canonical_key = method_target_path or f"unknown.{tool_name}"
            try:
                result, simulated = dispatch_sync(canonical_key, original, args, kwargs)
            except Exception as e:
                lifecycle.error_span_manual(span, parent_span_id, e)
                guard_exit(token)
                raise

            span.simulated = simulated
            if inspect.isgenerator(result):
                return wrap_generator(result, span, parent_span_id, lifecycle, token, guard_exit)

            span.result = result
            lifecycle.end_span_manual(span, parent_span_id)
            guard_exit(token)
            return result

        setattr(sync_wrapper, ATTR_WRAPPED, True)
        if method_target_path:
            setattr(sync_wrapper, ATTR_TARGET_PATHS, (method_target_path,))
            setattr(sync_wrapper, ATTR_DISPLAY_NAME, tool_name)
        return sync_wrapper


@overload
def link_agent(name: F) -> F: ...


@overload
def link_agent(
    name: str | None = None,
    *,
    entry_points: str | list[str] | None = None,
) -> Callable[[F], F]: ...  # @link_agent() or @link_agent("name")


def link_agent(
    name: str | Callable[..., Any] | None = None,
    *,
    entry_points: str | list[str] | None = None,
) -> Callable[..., Any]:
    """Decorator to register agent functions, classes, or objects with Tenro.

    When a Construct is active, the decorator records an agent span. Otherwise,
    the function/method executes normally. Set TENRO_LINKING_ENABLED=false to
    disable decorator wrapping and return the original target unchanged.

    Supports:
    - Sync and async functions
    - Classes with auto-detected entry methods (or explicit via entry_points=)
    - Framework objects (patches invoke/run methods)

    For classes, the decorator wraps ALL matching entry methods with a
    re-entrancy guard, so method delegation (e.g., invoke → stream)
    creates only one span.

    Can be used with or without parentheses:
    - @link_agent
    - @link_agent()
    - @link_agent("CustomName")

    Args:
        name: Agent name for the span. If None, uses function/class name.
            Can also be the target itself when used without parentheses.
        entry_points: For classes only. Explicit method name(s) to wrap.
            Can be a single string or list of strings. If None, auto-detects
            common entry methods (run, invoke, execute, call, stream, etc.).

    Returns:
        Decorated target that registers with active Construct.

    Raises:
        ValueError: If decorating a class and no entry method is found.

    Examples:
        >>> @link_agent
        ... def simple_agent(task: str) -> str:
        ...     return "done"
        >>>
        >>> @link_agent("PlannerBot")
        ... def plan_trip(destination: str) -> str:
        ...     return agent.run(destination)
        >>>
        >>> @link_agent("WriterAgent")
        ... class WriterAgent:
        ...     async def execute(self, prompt: str) -> str:
        ...         return "result"
        >>>
        >>> @link_agent("MultiEntry", entry_points=["run", "stream"])
        ... class MultiEntryAgent:
        ...     def run(self, task: str) -> str: ...
        ...     def stream(self, task: str) -> Iterator[str]: ...
    """
    resolved_name: str | None = None if callable(name) and not isinstance(name, str) else name

    def decorator(target: Any) -> Any:
        if not _is_linking_enabled():
            return target

        target_type = detect_target_type(target)
        agent_name: str = (
            resolved_name if resolved_name else getattr(target, "__name__", None) or str(target)
        )

        if target_type == TargetType.CLASS:
            return _decorate_agent_class(target, agent_name, entry_points)
        elif target_type == TargetType.FRAMEWORK_OBJECT:
            return _patch_agent_object(target, agent_name, entry_points)
        else:
            if entry_points is not None:
                raise TypeError(
                    f"@link_agent('{agent_name}', entry_points=...): "
                    f"entry_points is only valid for classes, not functions"
                )
            return _wrap_agent_function(target, agent_name)

    if callable(name) and not isinstance(name, str):
        # name is the target function when used as @link_agent without parens
        return cast("Callable[..., Any]", decorator(name))

    return decorator


@overload
def link_llm(provider: F) -> F: ...  # @link_llm without parentheses


@overload
def link_llm(
    provider: str | None = None,
    model: str | None = None,
) -> Callable[[F], F]: ...  # @link_llm() or @link_llm(Provider.OPENAI)


def link_llm(
    provider: str | Callable[..., Any] | None = None,
    model: str | None = None,
) -> Callable[..., Any]:
    """Decorator to mark functions as LLM call boundaries.

    Creates an LLMScope (transparent annotation span) when a Construct is active.
    HTTP interception will create LLMCall spans inside this scope. The scope
    captures caller info for debugging but is transparent for parent attribution.

    The provider can be specified explicitly or inferred automatically from HTTP
    interception (URL pattern matching) or simulation configuration.

    Set TENRO_LINKING_ENABLED=false to disable decorator wrapping and return the
    original function unchanged.

    Can be used with or without parentheses:
    - @link_llm
    - @link_llm()
    - @link_llm(Provider.OPENAI)
    - @link_llm(provider=Provider.OPENAI, model="gpt-4")

    Args:
        provider: LLM provider (e.g., Provider.OPENAI, Provider.ANTHROPIC), or
            the decorated function when used without parentheses. If None,
            provider is inferred from HTTP interception.
        model: Model identifier (e.g., "gpt-4", "claude-3").

    Returns:
        Decorated function that creates LLMScope when Construct is active.

    Examples:
        >>> @link_llm  # Provider inferred from HTTP call
        ... def call_llm(prompt: str) -> str:
        ...     return client.chat.completions.create(...)
        >>>
        >>> @link_llm(Provider.OPENAI, model="gpt-4")  # Explicit provider
        ... def call_openai(prompt: str) -> str:
        ...     return openai_client.chat.completions.create(...)
    """
    resolved_provider: str | None = (
        None if callable(provider) and not isinstance(provider, str) else provider
    )

    def decorator(func: F) -> F:
        if not _is_linking_enabled():
            return func

        if getattr(func, ATTR_WRAPPED, False):
            return func

        # Reject classes - link_llm is for functions only
        if inspect.isclass(func):
            from tenro.errors import TenroConfigError

            raise TenroConfigError(
                f"@link_llm cannot be applied to class '{func.__name__}'. "
                "Use @link_llm on functions only. "
                "For class-based LLM wrappers, use @link_tool or @link_agent instead."
            )

        sig = inspect.signature(func)
        caller_name = func.__name__
        caller_signature = f"{func.__qualname__}{sig}"
        try:
            file = inspect.getsourcefile(func) or inspect.getfile(func)
            line = inspect.getsourcelines(func)[1]
            caller_location: str | None = format_file_location(file, line)
        except (OSError, TypeError):
            caller_location = None

        # Generate target path for dispatch lookup
        target_path = f"{func.__module__}.{func.__qualname__}"

        # Stamp identity on original for bidirectional resolution
        _stamp_identity_on_original(func, target_path)

        if inspect.isasyncgenfunction(func):

            @wraps(func)
            async def asyncgen_wrapper(*args: Any, **kwargs: Any) -> Any:
                construct = get_active_construct()
                if not construct:
                    async for item in func(*args, **kwargs):
                        yield item
                    return

                # Create scope for the streaming LLM call
                scope = LLMScope(
                    id=str(uuid7()),
                    trace_id=str(uuid7()),
                    start_time=time.time(),
                    provider=resolved_provider,
                    model=model,
                    caller_name=caller_name,
                    caller_signature=caller_signature,
                    caller_location=caller_location,
                    input_data=args,
                    input_kwargs=kwargs,
                )
                lifecycle = construct._lifecycle
                parent_span_id = lifecycle.start_span_manual(scope)

                error: Exception | None = None
                collected_chunks: list[Any] = []
                try:
                    gen, simulated = await dispatch_asyncgen(target_path, func, args, kwargs)
                    if simulated:
                        # Create LLMCall span for tracking/verification
                        effective_model = get_simulation_model(target_path, model)
                        span = LLMCall(
                            id=str(uuid7()),
                            trace_id=str(uuid7()),
                            start_time=time.time(),
                            provider=resolved_provider or "custom",
                            messages=[],
                            response="",  # Will be updated after iteration
                            model=effective_model,
                            target_path=target_path,
                            llm_scope_id=scope.id,
                        )
                        span.simulated = True
                        span_parent_id = lifecycle.start_span_manual(span)
                        try:
                            async for item in gen:
                                collected_chunks.append(item)
                                yield item
                            # Update response with collected chunks
                            span.response = "".join(str(c) for c in collected_chunks)
                        finally:
                            lifecycle.end_span_manual(span, span_parent_id)
                    else:
                        async for item in gen:
                            collected_chunks.append(item)
                            yield item
                    scope.output_data = collected_chunks
                except Exception as e:
                    error = e
                    raise
                finally:
                    if error is not None:
                        lifecycle.error_span_manual(scope, parent_span_id, error)
                    else:
                        lifecycle.end_span_manual(scope, parent_span_id)

            setattr(asyncgen_wrapper, ATTR_WRAPPED, True)
            setattr(asyncgen_wrapper, ATTR_LINKED_TYPE, "llm")
            setattr(asyncgen_wrapper, ATTR_TARGET_PATHS, (target_path,))
            return asyncgen_wrapper  # type: ignore[return-value]

        if inspect.isgeneratorfunction(func):

            @wraps(func)
            def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                construct = get_active_construct()
                if not construct:
                    yield from func(*args, **kwargs)
                    return

                # Create scope for the streaming LLM call
                scope = LLMScope(
                    id=str(uuid7()),
                    trace_id=str(uuid7()),
                    start_time=time.time(),
                    provider=resolved_provider,
                    model=model,
                    caller_name=caller_name,
                    caller_signature=caller_signature,
                    caller_location=caller_location,
                    input_data=args,
                    input_kwargs=kwargs,
                )
                lifecycle = construct._lifecycle
                parent_span_id = lifecycle.start_span_manual(scope)

                error: Exception | None = None
                collected_chunks: list[Any] = []
                try:
                    gen, simulated = dispatch_gen(target_path, func, args, kwargs)
                    if simulated:
                        # Create LLMCall span for tracking/verification
                        effective_model = get_simulation_model(target_path, model)
                        span = LLMCall(
                            id=str(uuid7()),
                            trace_id=str(uuid7()),
                            start_time=time.time(),
                            provider=resolved_provider or "custom",
                            messages=[],
                            response="",  # Will be updated after iteration
                            model=effective_model,
                            target_path=target_path,
                            llm_scope_id=scope.id,
                        )
                        span.simulated = True
                        span_parent_id = lifecycle.start_span_manual(span)
                        try:
                            for item in gen:
                                collected_chunks.append(item)
                                yield item
                            # Update response with collected chunks
                            span.response = "".join(str(c) for c in collected_chunks)
                        finally:
                            lifecycle.end_span_manual(span, span_parent_id)
                    else:
                        for item in gen:
                            collected_chunks.append(item)
                            yield item
                    scope.output_data = collected_chunks
                except Exception as e:
                    error = e
                    raise
                finally:
                    if error is not None:
                        lifecycle.error_span_manual(scope, parent_span_id, error)
                    else:
                        lifecycle.end_span_manual(scope, parent_span_id)

            setattr(gen_wrapper, ATTR_WRAPPED, True)
            setattr(gen_wrapper, ATTR_LINKED_TYPE, "llm")
            setattr(gen_wrapper, ATTR_TARGET_PATHS, (target_path,))
            return gen_wrapper  # type: ignore[return-value]

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                construct = get_active_construct()
                if not construct:
                    return await func(*args, **kwargs)

                # Create scope and check for dispatch
                scope = LLMScope(
                    id=str(uuid7()),
                    trace_id=str(uuid7()),
                    start_time=time.time(),
                    provider=resolved_provider,
                    model=model,
                    caller_name=caller_name,
                    caller_signature=caller_signature,
                    caller_location=caller_location,
                    input_data=args,
                    input_kwargs=kwargs,
                )
                with construct._lifecycle.start_span(scope):
                    # Check for simulation via dispatch
                    result, simulated = await dispatch_async(target_path, func, args, kwargs)
                    if simulated:
                        # Create LLMCall span for tracking/verification
                        effective_model = get_simulation_model(target_path, model)
                        span = LLMCall(
                            id=str(uuid7()),
                            trace_id=str(uuid7()),
                            start_time=time.time(),
                            provider=resolved_provider or "custom",
                            messages=[],  # Dispatch doesn't capture messages
                            response=result if isinstance(result, str) else str(result),
                            model=effective_model,
                            target_path=target_path,
                            llm_scope_id=scope.id,  # Link to enclosing scope
                        )
                        span.simulated = True
                        with construct._lifecycle.start_span(span):
                            scope.output_data = result
                            return result
                    # Not simulated - result is already computed by dispatch
                    scope.output_data = result
                    return result

            setattr(async_wrapper, ATTR_WRAPPED, True)
            setattr(async_wrapper, ATTR_LINKED_TYPE, "llm")
            setattr(async_wrapper, ATTR_TARGET_PATHS, (target_path,))
            return async_wrapper  # type: ignore[return-value]
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                construct = get_active_construct()
                if not construct:
                    return func(*args, **kwargs)

                # Create scope and check for dispatch
                scope = LLMScope(
                    id=str(uuid7()),
                    trace_id=str(uuid7()),
                    start_time=time.time(),
                    provider=resolved_provider,
                    model=model,
                    caller_name=caller_name,
                    caller_signature=caller_signature,
                    caller_location=caller_location,
                    input_data=args,
                    input_kwargs=kwargs,
                )
                with construct._lifecycle.start_span(scope):
                    # Check for simulation via dispatch
                    result, simulated = dispatch_sync(target_path, func, args, kwargs)
                    if simulated:
                        # Create LLMCall span for tracking/verification
                        effective_model = get_simulation_model(target_path, model)
                        span = LLMCall(
                            id=str(uuid7()),
                            trace_id=str(uuid7()),
                            start_time=time.time(),
                            provider=resolved_provider or "custom",
                            messages=[],  # Dispatch doesn't capture messages
                            response=result if isinstance(result, str) else str(result),
                            model=effective_model,
                            target_path=target_path,
                            llm_scope_id=scope.id,  # Link to enclosing scope
                        )
                        span.simulated = True
                        with construct._lifecycle.start_span(span):
                            scope.output_data = result
                            return result
                    # Not simulated - result is already computed by dispatch
                    scope.output_data = result
                    return result

            setattr(sync_wrapper, ATTR_WRAPPED, True)
            setattr(sync_wrapper, ATTR_LINKED_TYPE, "llm")
            setattr(sync_wrapper, ATTR_TARGET_PATHS, (target_path,))
            return sync_wrapper  # type: ignore[return-value]

    if callable(provider) and not isinstance(provider, str):
        return decorator(provider)

    return decorator


@overload
def link_tool(name: F) -> F: ...  # @link_tool without parentheses


@overload
def link_tool(
    name: str | None = None,
    *,
    entry_points: str | list[str] | None = None,
) -> Callable[[F], F]: ...  # @link_tool("name")


def link_tool(
    name: str | Callable[..., Any] | None = None,
    *,
    entry_points: str | list[str] | None = None,
) -> Callable[..., Any]:
    """Decorator to register tool functions, classes, or objects with Tenro.

    When a Construct is active, the decorator records a tool span. Otherwise,
    the function executes normally. Set TENRO_LINKING_ENABLED=false to disable
    decorator wrapping and return the original target unchanged.

    At decoration time (import), registers tool to GlobalDeclaredRegistry
    for attack surface tracking and coverage calculation.

    Supports:
    - Sync and async functions
    - Classes with auto-detected entry methods (or explicit via entry_points=)
    - Framework objects (patches invoke/run methods)

    Can be used with or without parentheses:
    - @link_tool
    - @link_tool()
    - @link_tool("CustomName")

    Args:
        name: Tool name for the span. If None, uses function/class name.
            Can also be the target itself when used without parentheses.
        entry_points: For classes only. Explicit method name(s) to wrap.
            Can be a single string or list of strings. If None, auto-detects
            common entry methods (run, invoke, execute, call, etc.).

    Returns:
        Decorated target that registers with active Construct.

    Examples:
        >>> @link_tool
        ... def simple_tool(query: str) -> str:
        ...     return "result"
        >>>
        >>> @link_tool("search")
        ... def search(query: str) -> list[str]:
        ...     return ["result1", "result2"]
        >>>
        >>> @link_tool("calculator")
        ... class Calculator:
        ...     def invoke(self, expr: str) -> int:
        ...         return eval(expr)
        >>>
        >>> @link_tool("multi_tool", entry_points=["search", "fetch"])
        ... class MultiTool:
        ...     def search(self, q: str) -> list[str]: ...
        ...     def fetch(self, url: str) -> str: ...
    """
    resolved_name: str | None = None if callable(name) and not isinstance(name, str) else name

    def decorator(target: Any) -> Any:
        if not _is_linking_enabled():
            return target

        target_type = detect_target_type(target)
        tool_name: str = (
            resolved_name if resolved_name else getattr(target, "__name__", None) or str(target)
        )

        if target_type == TargetType.CLASS:
            return _decorate_tool_class(target, tool_name, entry_points)
        elif target_type == TargetType.FRAMEWORK_OBJECT:
            return _patch_tool_object(target, tool_name, entry_points)
        else:
            if entry_points is not None:
                raise TypeError(
                    f"@link_tool('{tool_name}', entry_points=...): "
                    f"entry_points is only valid for classes, not functions"
                )
            return _wrap_tool_function(target, tool_name)

    if callable(name) and not isinstance(name, str):
        return cast("Callable[..., Any]", decorator(name))

    return decorator


__all__ = [
    "_get_active_construct",
    "_set_active_construct",
    "link_agent",
    "link_llm",
    "link_tool",
]
