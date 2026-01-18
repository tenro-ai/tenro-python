# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Target resolution utilities for simulation.

Handles validation and resolution of simulation targets (functions, methods, paths).
"""

from __future__ import annotations

import functools
import inspect
import sys
from typing import Any

from tenro._construct.simulate.types import OperationType, SimulationType

# Identity key attribute stamped on originals at link-time for bidirectional resolution
_IDENTITY_KEY_ATTR = "_tenro_identity_key"


def _get_identity_key(target: Any) -> str | None:
    """Get identity key from a target or its unwrapped form.

    Supports bidirectional lookup: both wrapper and original can resolve
    to the same canonical path via the stamped identity key.

    Args:
        target: Target to check (wrapper or original function).

    Returns:
        Canonical target path if identity key found, None otherwise.
    """
    # Try direct attribute first - fastest path for undecorated callables
    identity: str | None = getattr(target, _IDENTITY_KEY_ATTR, None)
    if identity is not None:
        return identity

    # Fall back to unwrapped form for decorated/wrapped callables to reach original
    try:
        unwrapped = inspect.unwrap(target)
        if unwrapped is not target:
            unwrapped_identity: str | None = getattr(unwrapped, _IDENTITY_KEY_ATTR, None)
            if unwrapped_identity is not None:
                return unwrapped_identity
    except (ValueError, TypeError):
        pass

    return None


def _find_longest_module_prefix(parts: list[str]) -> tuple[Any, int]:
    """Find longest imported module prefix from path parts.

    Supports namespace packages where parent modules may not be registered.

    Args:
        parts: Path split by dots.

    Returns:
        Tuple of (module, end_index) or (None, 0) if not found.
    """
    module = None
    module_end_idx = 0
    for i in range(1, len(parts)):
        potential_module_name = ".".join(parts[:i])
        if potential_module_name in sys.modules:
            module = sys.modules[potential_module_name]
            module_end_idx = i
    return module, module_end_idx


def _walk_to_container(start: Any, path_parts: list[str]) -> Any | None:
    """Walk from start object through path parts to find container.

    Args:
        start: Starting object (typically a module).
        path_parts: Attribute names to traverse.

    Returns:
        Container object, or None if path is invalid.
    """
    container = start
    for part in path_parts:
        if not hasattr(container, part):
            return None
        container = getattr(container, part)
    return container


def _coerce_target_paths(obj: Any) -> tuple[str, ...] | None:
    """Extract and validate _tenro_target_paths from an object.

    Args:
        obj: Object to check for _tenro_target_paths attribute.

    Returns:
        Tuple of path strings, or None if attribute not present/empty.

    Raises:
        TypeError: If _tenro_target_paths is a str instead of tuple.
    """
    if not hasattr(obj, "_tenro_target_paths"):
        return None
    raw = obj._tenro_target_paths
    if isinstance(raw, str):
        raise TypeError(
            f"_tenro_target_paths must be tuple[str, ...], not str. "
            f"Got: {raw!r} on {type(obj).__name__}"
        )
    if not raw:
        return None
    return tuple(str(p) for p in raw)


class _TargetValidator:
    """Validates targets for simulate/verify operations."""

    def __init__(
        self,
        simulation_type: SimulationType,
        operation: OperationType = OperationType.SIMULATE,
    ) -> None:
        self.simulation_type = simulation_type
        self.operation = operation

    @property
    def _method_name(self) -> str:
        return f"{self.operation}_{self.simulation_type}"

    @property
    def _decorator_name(self) -> str:
        return f"@link_{self.simulation_type}"

    def validate(self, target: Any) -> None:
        """Validate target can be used for the operation.

        Args:
            target: Target to validate.

        Raises:
            ValueError: If target is invalid for the operation.
        """
        if isinstance(target, str):
            validate_string_path(target)
            return

        name = _get_unpatchable_callable_name(target)
        if name:
            raise ValueError(
                f"{self._method_name} target '{name}' is a builtin/C-implemented function. "
                f"Pass a pure-Python function/method or use {self._decorator_name} on a wrapper."
            )

        # Check for local definitions that can't be patched
        is_linked = hasattr(target, "_tenro_target_paths") or hasattr(target, "_tenro_linked_type")
        is_simulate = self.operation == OperationType.SIMULATE
        skip_locals_check = inspect.isclass(target) or is_linked
        if is_simulate and not skip_locals_check:
            # First check __qualname__ directly for local functions
            qualname = getattr(target, "__qualname__", "")
            if "<locals>" in qualname:
                raise ValueError(
                    f"Cannot use local definition as target: '{qualname}' contains "
                    f"'<locals>' and cannot be patched. Move the definition to module level "
                    f"or use {self._decorator_name}."
                )

            # Check resolved paths for nested class methods
            is_tool = self.simulation_type == SimulationType.TOOL
            try:
                paths = resolve_all_target_paths(target, is_tool=is_tool)
            except ValueError:
                # Path resolution may fail for some targets; other validation will catch it
                paths = ()

            for path in paths:
                if "<locals>" in path:
                    raise ValueError(
                        f"Cannot use local definition as target: path '{path}' contains "
                        f"'<locals>' and cannot be patched. Move the definition to module level."
                    )

        if self.simulation_type == SimulationType.LLM:
            self._validate_llm_target(target)
            return

        if callable(target):
            return

        from tenro.linking.constants import AGENT_ENTRY_METHODS, TOOL_ENTRY_METHODS

        all_entry_methods = AGENT_ENTRY_METHODS | TOOL_ENTRY_METHODS
        if any(
            hasattr(target, m) and callable(getattr(target, m, None)) for m in all_entry_methods
        ):
            return

        raise ValueError(
            f"{self._method_name} target {type(target).__name__!r} has no recognized "
            f"entry method (run, execute, invoke, __call__). Pass the method directly "
            f"(e.g., obj.run) or use entry_points= in {self._decorator_name}."
        )

    def _validate_llm_target(self, target: Any) -> None:
        if target is None:
            return
        if inspect.isclass(target):
            raise ValueError(
                f"{self._method_name} target {target.__name__!r} is a class. "
                f"Pass a method (e.g., {target.__name__}.method) or use "
                f"{self._decorator_name} on a function."
            )
        if inspect.isfunction(target) or inspect.ismethod(target):
            return
        if callable(target):
            raise ValueError(
                f"{self._method_name} target {type(target).__name__!r} is a callable instance. "
                f"Pass the method directly (e.g., obj.__call__) or use "
                f"{self._decorator_name} on a function."
            )
        raise ValueError(
            f"{self._method_name} target {type(target).__name__!r} is not callable. "
            f"Pass a function, method, or use {self._decorator_name}."
        )


def validate_and_resolve_target(target: Any, expected_type: str) -> tuple[str, ...]:
    """Validate callable target and return ALL paths for simulation.

    Accepts string paths, classes, unbound class methods, or decorated functions.
    Validates target is patchable, then resolves to paths.

    Args:
        target: Target to validate (string path or callable object).
        expected_type: Expected link type ("tool", "agent", "llm").

    Returns:
        Tuple of string paths to use for simulation.

    Raises:
        ValueError: If target is not patchable (builtin, C-implemented, etc.).
    """
    sim_type = {
        "agent": SimulationType.AGENT,
        "tool": SimulationType.TOOL,
        "llm": SimulationType.LLM,
    }[expected_type]
    validate_target_is_patchable(target, simulation_type=sim_type)
    is_tool = expected_type == "tool"
    return resolve_all_target_paths(target, is_tool=is_tool)


def _get_unpatchable_callable_name(target: object) -> str | None:
    """Return name for builtin/C-implemented callables that are unsafe to patch."""
    if inspect.isbuiltin(target) or inspect.ismethoddescriptor(target):
        return (
            getattr(target, "__qualname__", None)
            or getattr(target, "__name__", None)
            or str(target)
        )
    # Catch remaining C-implemented callables/descriptors (global blast radius)
    if getattr(target, "__module__", None) == "builtins":
        return (
            getattr(target, "__qualname__", None)
            or getattr(target, "__name__", None)
            or str(target)
        )
    return None


def _compute_class_path(cls: type, is_tool: bool = False) -> str:
    """Compute path for a class, preferring entry methods if available.

    Args:
        cls: The class to compute path for.
        is_tool: If True, use tool entry precedence; otherwise agent precedence.
    """
    from tenro.linking.metadata import find_entry_methods, find_tool_entry_methods

    module = getattr(cls, "__module__", "")
    qualname = getattr(cls, "__qualname__", cls.__name__)

    methods = find_tool_entry_methods(cls) if is_tool else find_entry_methods(cls)
    if methods:
        return f"{module}.{qualname}.{methods[0]}"
    return f"{module}.{qualname}"


def resolve_target_for_verification(target: Any) -> tuple[str, ...]:
    """Resolve a target to ALL target_path strings for verification filtering.

    Uses _tenro_target_paths for decorated targets, computes path for undecorated.
    Returns all paths so verification can match any entry method.

    Args:
        target: Target to resolve (string or callable).

    Returns:
        Tuple of target_path strings to use for filtering.
    """
    return resolve_all_target_paths(target)


def _resolve_bound_method_paths(target: Any) -> tuple[str, ...]:
    """Resolve bound method to target paths.

    Checks metadata on __func__, unwrapped __func__, then instance's class.
    Falls back to qualname-based resolution if no metadata found.

    Args:
        target: Bound method to resolve.

    Returns:
        Tuple of target_path strings.
    """
    func = target.__func__
    method_name = func.__name__

    paths = _coerce_target_paths(func)
    if paths:
        return paths

    unwrapped = inspect.unwrap(func)
    if unwrapped is not func:
        paths = _coerce_target_paths(unwrapped)
        if paths:
            return paths

    instance_class = type(target.__self__)
    paths = _coerce_target_paths(instance_class)
    if paths:
        matching = tuple(p for p in paths if p.endswith(f".{method_name}"))
        if matching:
            return matching

    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", str(func)))
    path = f"{module}.{qualname}" if module else qualname
    return (path,)


def resolve_all_target_paths(target: Any, is_tool: bool = False) -> tuple[str, ...]:
    """Resolve a target to ALL target_path strings for verification/simulation.

    For classes/objects with multiple entry methods, returns all paths.
    This enables verify_agent(MyClass) to match calls to ANY entry method.

    Args:
        target: Target to resolve (string or callable).
        is_tool: If True, use tool entry precedence for undecorated classes.

    Returns:
        Tuple of target_path strings. Empty tuple for non-resolvable targets.

    Raises:
        ValueError: If target is a functools.partial or local function.
    """
    if isinstance(target, str):
        # Accept string paths as-is for verification (display names may not have dots)
        # Validation for simulation is done in orchestrator.simulate_tool/simulate_agent
        return (target,)

    # Try identity key first for bidirectional resolution (wrapper ↔ original)
    identity = _get_identity_key(target)
    if identity is not None:
        return (identity,)

    paths = _coerce_target_paths(target)
    if paths:
        return paths

    if isinstance(target, functools.partial):
        raise ValueError(
            "Cannot use functools.partial as target: it captures a direct reference "
            "to the underlying function, bypassing module-level patching (Import Trap). "
            "Pass the underlying function directly."
        )

    if isinstance(target, type):
        return _compute_all_method_paths(target, is_tool=is_tool)

    if inspect.isfunction(target):
        unwrapped = inspect.unwrap(target)
        if unwrapped is not target:
            paths = _coerce_target_paths(unwrapped)
            if paths:
                return paths
        module = getattr(target, "__module__", None)
        qualname = getattr(target, "__qualname__", getattr(target, "__name__", str(target)))
        if "<locals>" in qualname:
            raise ValueError(
                f"Cannot use local function '{qualname}' as target: nested functions "
                f"have '<locals>' in their path and cannot be patched. "
                f"Move the function to module level or use @link_agent/@link_tool decorator."
            )
        path = f"{module}.{qualname}" if module else qualname
        return (path,)

    if inspect.ismethod(target):
        return _resolve_bound_method_paths(target)

    target_class = type(target)
    if target_class is not type and target_class.__module__ != "builtins":
        paths = _coerce_target_paths(target_class)
        if paths:
            return paths
        return _compute_all_method_paths(target_class, is_tool=is_tool)

    return ()


def _compute_all_method_paths(cls: type, is_tool: bool = False) -> tuple[str, ...]:
    """Compute paths for all entry methods of an undecorated class."""
    from tenro.linking.metadata import (
        _is_explicitly_defined,
        find_entry_methods,
        find_tool_entry_methods,
    )

    module = getattr(cls, "__module__", "")
    qualname = getattr(cls, "__qualname__", cls.__name__)

    methods = find_tool_entry_methods(cls) if is_tool else find_entry_methods(cls)
    if methods:
        return tuple(f"{module}.{qualname}.{m}" for m in methods)

    # Fallback: check for __call__ if no standard entry methods found
    if _is_explicitly_defined(cls, "__call__"):
        return (f"{module}.{qualname}.__call__",)

    # Reject classes without entry methods (matches string path validation behavior)
    raise ValueError(
        f"Class '{cls.__name__}' has no entry methods to simulate. "
        f"Expected one of: execute, run, invoke, __call__, etc. "
        f"Use @link_agent or @link_tool to decorate specific methods, "
        f"or pass a method reference directly (e.g., {cls.__name__}.your_method)."
    )


def validate_target_is_patchable(
    target: Any,
    simulation_type: SimulationType = SimulationType.AGENT,
    operation: OperationType = OperationType.SIMULATE,
) -> None:
    """Validate target can be used for simulate/verify operations.

    Args:
        target: Target to validate.
        simulation_type: Type of target (agent, tool, or llm).
        operation: Type of operation (simulate or verify).

    Raises:
        ValueError: If target is invalid for the operation.
    """
    _TargetValidator(simulation_type, operation).validate(target)


def validate_string_path(path: str) -> None:
    """Validate a string path is a valid dotted path.

    String paths must contain at least one dot (e.g., 'module.function').
    Bare names like 'search' are rejected - use a callable reference instead.

    Args:
        path: Dotted path to validate.

    Raises:
        ValueError: If path has no dots or resolves to a class.
    """
    if "." not in path:
        raise ValueError(
            f"Path must contain at least one dot, got: {path!r}. "
            f"Use a function/method reference instead of a bare string name."
        )

    parts = path.split(".")
    module, module_end_idx = _find_longest_module_prefix(parts)

    if module is None:
        return

    container = _walk_to_container(module, parts[module_end_idx:-1])
    if container is None:
        return

    target_name = parts[-1]
    if hasattr(container, target_name):
        target = getattr(container, target_name)
        name = _get_unpatchable_callable_name(target)
        if name:
            raise ValueError(
                f"Invalid path '{path}': resolves to builtin or descriptor '{name}'. "
                "These targets are not safely patchable."
            )
        if isinstance(target, type):
            raise ValueError(
                f"Invalid path '{path}': resolves to a class. "
                f"Use full method path (e.g., '{path}.execute' or '{path}.invoke') "
                f"or pass the class object directly."
            )


def parse_dotted_path(path: str) -> tuple[Any, str]:
    """Parse dotted path into (container_object, attribute_name) for patching.

    Handles both module-level functions and class methods:
    - `module.func` -> (module, "func")
    - `module.Class.method` -> (Class, "method")
    - `module.Outer.Inner.method` -> (Inner, "method")

    Uses longest imported module prefix to support nested modules like
    `langchain.chat_models.ChatOpenAI.predict` where only `langchain.chat_models`
    is in sys.modules.

    Args:
        path: Dotted path to parse (e.g., "langchain.chat_models.ChatOpenAI.predict").

    Returns:
        Tuple of (container_object, attribute_name) for patching.

    Raises:
        ValueError: If no imported module prefix found.
        AttributeError: If path is invalid.
    """
    if "." not in path:
        raise ValueError(f"Path must contain at least one dot, got: {path}")

    parts = path.split(".")
    module, module_end_idx = _find_longest_module_prefix(parts)

    if module is None:
        raise ValueError(
            f"No imported module prefix found for '{path}'. "
            f"Import the module containing the target before simulating."
        )

    container = module
    for part in parts[module_end_idx:-1]:
        if not hasattr(container, part):
            raise AttributeError(
                f"'{container}' has no attribute '{part}' (while parsing path '{path}')"
            )
        container = getattr(container, part)

    target_name = parts[-1]
    if hasattr(container, target_name):
        target = getattr(container, target_name)
        name = _get_unpatchable_callable_name(target)
        if name:
            raise ValueError(
                f"Invalid path '{path}': resolves to builtin/descriptor '{name}'. "
                "These targets are not safely patchable."
            )
        if isinstance(target, type):
            raise ValueError(
                f"Invalid path '{path}': resolves to a class. "
                f"Use full method path (e.g., '{path}.execute' or '{path}.invoke') "
                f"or pass the class object directly."
            )

    return container, target_name
