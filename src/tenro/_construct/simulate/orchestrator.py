# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Simulation orchestrator for Construct testing harness.

Manages simulation rules, patching, and lifecycle integration for
tool, agent, and LLM simulations.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from uuid_utils import uuid7

from tenro._construct.http.builders import ProviderSchemaFactory
from tenro._construct.simulate.helpers import (
    normalize_result_sequence,
    validate_simulation_params,
)
from tenro._construct.simulate.llm import (
    ToolCall as SimToolCall,
)
from tenro._construct.simulate.llm import (
    resolve_provider_from_target,
    should_use_http_interception,
    validate_llm_simulation_params,
)
from tenro._construct.simulate.llm.tool_call import normalize_tool_calls
from tenro._construct.simulate.response_parser import contributes_tool_calls, parse_response_item
from tenro._construct.simulate.rule import SimulationRule
from tenro._construct.simulate.target_resolution import (
    parse_dotted_path,
    resolve_all_target_paths,
    validate_and_resolve_target,
    validate_string_path,
    validate_target_is_patchable,
)
from tenro._construct.simulate.tracker import SimulationTracker, ToolAgentTracker
from tenro._construct.simulate.types import ResponsesInput, SimulationType
from tenro._core import context
from tenro._core.response_types import ProviderResponse
from tenro._core.spans import LLMCall
from tenro.linking.metadata import get_display_name
from tenro.llm_response import LLMResponse, RawLLMResponse
from tenro.tool_calls import ToolCall

if TYPE_CHECKING:
    from tenro._construct.http import HttpInterceptor
    from tenro._core.lifecycle_manager import LifecycleManager


def _is_class_scoped_callable(target: Any) -> bool:
    """Check if callable has a class owner (method or unbound method).

    Args:
        target: Any object to check.

    Returns:
        True for unbound methods (Class.method) or bound methods (instance.method).
        False for module-level functions, local/nested functions, or non-callables.
    """
    # Bound methods always have __self__ and underlying __func__
    if inspect.ismethod(target):
        return True

    # Unbound methods are functions with class-qualified names
    if inspect.isfunction(target):
        qualname = getattr(target, "__qualname__", "")
        return "." in qualname and "<locals>" not in qualname

    return False


def _is_module_level_function(target: Any) -> bool:
    """Check if target is a module-level function.

    Module-level functions are not class methods and have no dots in qualname.
    """
    return inspect.isfunction(target) and "." not in getattr(target, "__qualname__", "")


def _resolve_callable_from_container(container: Any, attr_name: str) -> Any | None:
    """Resolve callable from container, handling descriptors correctly.

    For class containers, gets raw descriptor from MRO to avoid the classmethod
    bound-method identity issue (getattr returns new bound method each time).

    Args:
        container: Module, class, or instance containing the attribute.
        attr_name: Name of the attribute to resolve.

    Returns:
        The underlying callable, or None if not found.
    """
    if inspect.isclass(container):
        # MRO lookup avoids descriptor protocol triggering new bound methods
        for klass in container.__mro__:
            if attr_name in klass.__dict__:
                raw = klass.__dict__[attr_name]
                if isinstance(raw, classmethod):
                    return raw.__func__
                if isinstance(raw, staticmethod):
                    return raw.__func__
                return raw
        return None
    # Module or instance - use getattr
    return getattr(container, attr_name, None)


def _resolve_string_target(target: str | Any) -> Any:
    """Resolve string path to actual callable for linked/registered check.

    For string targets like "module.Class.method", resolves to the actual
    method/function object to check if it's linked or registered.

    Uses descriptor-aware resolution to handle classmethod/staticmethod
    (getattr returns new bound method for classmethod on each access).

    Returns the original target if not a string or if resolution fails.
    """
    if not isinstance(target, str):
        return target

    try:
        from tenro._construct.simulate.target_resolution import parse_dotted_path

        container, attr_name = parse_dotted_path(target)
        # Descriptor-aware: getattr creates new bound method each call for classmethod
        resolved = _resolve_callable_from_container(container, attr_name)
        return resolved if resolved is not None else target
    except (ValueError, AttributeError):
        return target


def _get_linked_type(target: Any) -> str | None:
    """Get the linked type of a target.

    Args:
        target: Function, method, or class to check.

    Returns:
        "agent", "tool", or "llm" for decorated targets; None if not linked.
    """
    if hasattr(target, "_tenro_linked_type"):
        linked_type: str = target._tenro_linked_type
        return linked_type

    # Identity key on originals points to wrapper; resolve to get linked type
    identity_key = getattr(target, "_tenro_identity_key", None)
    if identity_key:
        from tenro._construct.simulate.target_resolution import parse_dotted_path

        container, attr_name = parse_dotted_path(identity_key)
        wrapper = _resolve_callable_from_container(container, attr_name)
        if wrapper and hasattr(wrapper, "_tenro_linked_type"):
            wrapper_type: str = wrapper._tenro_linked_type
            return wrapper_type

    # For class-scoped callables, check if the owning class is decorated
    if _is_class_scoped_callable(target):
        # For bound methods, check the owner class
        if inspect.ismethod(target):
            self_obj = target.__self__
            # For classmethods, __self__ IS the class; for instance methods, it's the instance
            owner_class = self_obj if inspect.isclass(self_obj) else self_obj.__class__
            if hasattr(owner_class, "_tenro_linked_type"):
                return owner_class._tenro_linked_type  # type: ignore[no-any-return]
            return None

        # For unbound methods (functions), resolve class via qualname
        qualname = getattr(target, "__qualname__", "")
        if "." in qualname:
            class_path = qualname.rsplit(".", 1)[0]
            module = getattr(target, "__module__", None)
            if module:
                import sys

                mod = sys.modules.get(module)
                if mod:
                    # Traverse nested class hierarchy (Outer.Inner)
                    # to find the owning class and check if decorated
                    obj: Any = mod
                    for part in class_path.split("."):
                        obj = getattr(obj, part, None)
                        if obj is None:
                            break
                    if obj and hasattr(obj, "_tenro_linked_type"):
                        return obj._tenro_linked_type  # type: ignore[no-any-return]

    return None


def _is_linked_target(target: Any) -> bool:
    """Check if target is linked (decorated with @link_agent/@link_tool/@link_llm).

    For unbound methods from decorated classes, checks the owning class.
    Also recognizes original callables stamped with identity key at link-time.
    This ensures dispatch-based simulation is used instead of setattr patching.
    """
    # Direct identity key indicates explicitly linked callable
    if hasattr(target, "_tenro_identity_key"):
        return True

    return _get_linked_type(target) is not None


def _is_linked_type(target: Any, expected_type: str) -> bool:
    """Check if target is linked with a specific type.

    Args:
        target: The target to check.
        expected_type: Expected linked type ("agent", "tool", or "llm").

    Returns:
        True if target is linked with the expected type.
    """
    return _get_linked_type(target) == expected_type


def _is_registered_target(target: Any) -> bool:
    """Check if target is registered via tenro.register() for capture-safe simulation.

    Handles bound methods (from classmethods) by normalizing to __func__ before
    checking the registry.
    """
    if not callable(target) or inspect.isclass(target):
        return False

    from tenro.simulate._register import is_registered

    # Bound methods from classmethod need __func__ to match registry key
    normalized = target.__func__ if inspect.ismethod(target) else target
    return is_registered(normalized)


def _is_generator_target(target: Any) -> bool:
    """Check if target is a generator or async generator function.

    Handles:
    - String targets: resolves to callable first
    - Callable targets (functions/methods): checks __wrapped__ then target itself
    - Objects with entry methods (framework objects): checks entry methods
    """
    # Resolve string targets to actual callable
    if isinstance(target, str):
        resolved = _resolve_string_target(target)
        if resolved is not None:
            target = resolved

    # For callable targets, check for wrapped original
    if callable(target) and not inspect.isclass(target):
        original = getattr(target, "__wrapped__", None)
        if original is None:
            original = getattr(target, "_tenro_original", target)
        return inspect.isgeneratorfunction(original) or inspect.isasyncgenfunction(original)

    # For non-callable objects (framework objects), check entry methods
    if hasattr(target, "_tenro_linked_type"):
        from tenro.linking.constants import AGENT_ENTRY_METHODS, TOOL_ENTRY_METHODS

        entry_methods = AGENT_ENTRY_METHODS | TOOL_ENTRY_METHODS
        for method_name in entry_methods:
            method = getattr(target, method_name, None)
            if method is not None and callable(method):
                original = getattr(method, "__wrapped__", method)
                if inspect.isgeneratorfunction(original) or inspect.isasyncgenfunction(original):
                    return True

    return False


def _detect_target_type(target: Any) -> str:
    """Detect the type of a target for diagnostic reporting.

    Args:
        target: The simulation target.

    Returns:
        String describing the target type (function, method, class, builtin, etc.).
    """
    if isinstance(target, str):
        return "string_path"
    if inspect.isclass(target):
        return "class"
    if inspect.isbuiltin(target):
        return "builtin"
    if inspect.ismethod(target):
        return "method"
    if inspect.isfunction(target):
        qualname = getattr(target, "__qualname__", "")
        if "." in qualname and "<locals>" not in qualname:
            return "method"
        return "function"
    if callable(target):
        return "callable"
    return "unknown"


def _normalize_callable(obj: Any) -> Any:
    """Return stable underlying callable for identity/registration checks.

    Classmethods return a new bound method on each attribute access, breaking
    identity-based registry checks. This helper extracts the underlying function.

    Args:
        obj: Any callable object.

    Returns:
        The underlying function for bound methods, or the object unchanged.
    """
    if inspect.ismethod(obj):
        return obj.__func__
    return obj


def _raise_non_linked_error(
    target_path: str,
    sim_type: str,
    recommended_fix: str,
    failure_reason: str,
    target_type: str = "function",
) -> None:
    """Raise TenroSimulationSetupError for non-linked target.

    Args:
        target_path: Display name or path for the target.
        sim_type: Simulation type for error messages ("tool" or "agent").
        recommended_fix: Suggested fix for the error message.
        failure_reason: Explanation of why the target cannot be simulated.
        target_type: Type of the target (function, method, class, etc.).
    """
    from tenro.errors.simulation import (
        SimulationDiagnostic,
        TenroSimulationSetupError,
    )

    diagnostic = SimulationDiagnostic(
        target_path=target_path,
        target_type=target_type,
        is_linked=False,
        failure_reason=failure_reason,
        recommended_fix=recommended_fix,
    )
    msg = (
        f"Cannot simulate non-linked {sim_type} target '{target_path}'.\n\n"
        f"Reason: {diagnostic.failure_reason}\n"
        f"Fix: {diagnostic.recommended_fix}"
    )
    raise TenroSimulationSetupError(msg, diagnostic=diagnostic)


def _validate_string_path_target(path: str, sim_type: str, recommended_fix: str) -> None:
    """Validate that a string path resolves to a simulatable target.

    A target is simulatable if it's linked (@link_*) or registered (tenro.register()).
    String paths must resolve to simulatable targets for capture-safe simulation.
    """
    from tenro._construct.simulate.target_resolution import parse_dotted_path

    try:
        container, attr_name = parse_dotted_path(path)
    except (ValueError, AttributeError):
        # Path doesn't resolve - let other validation handle this
        return

    # Use descriptor-aware resolution for classes (handles classmethod/staticmethod)
    resolved_target = _resolve_callable_from_container(container, attr_name)
    if resolved_target is None:
        return

    # Normalize for identity checks (handles bound methods from classmethods)
    normalized = _normalize_callable(resolved_target)

    # Skip if target is linked or registered (capture-safe)
    if _is_linked_target(normalized) or _is_registered_target(normalized):
        return

    # For classes, check if the class is linked
    if inspect.isclass(resolved_target) and hasattr(resolved_target, "_tenro_linked_type"):
        return

    # Non-linked target via string path - NOT supported
    _raise_non_linked_error(
        path,
        sim_type,
        recommended_fix,
        failure_reason=(
            "String path resolves to non-linked target. "
            "Setattr patching is not supported at simulate() time."
        ),
        target_type=_detect_target_type(resolved_target),
    )


def _resolve_display_name(target: Any, fallback_path: str) -> str | None:
    """Resolve display_name from target, falling back to path-derived name.

    For decorated targets, uses the decorator's display_name. For undecorated
    targets, derives name from the last segment of the path.

    Args:
        target: The simulation target (class, method, or string).
        fallback_path: Path string to derive name from if target has no display_name.

    Returns:
        The resolved display name, or None if neither is available.
    """
    display_name = get_display_name(target)

    # For methods, try to get display_name from the owning class
    if display_name is None and callable(target):
        qualname = getattr(target, "__qualname__", "")
        if "." in qualname:
            # Method - try to find the class via __self__ or by resolving the class name
            owner = getattr(target, "__self__", None)
            if owner is not None:
                # Bound method - check the instance's class
                display_name = get_display_name(type(owner))
            else:
                # Unbound method - check for class reference
                for attr in ("__objclass__", "__class__"):
                    cls = getattr(target, attr, None)
                    if cls is not None and isinstance(cls, type):
                        display_name = get_display_name(cls)
                        if display_name:
                            break

    # Fall back to path-derived name
    if display_name is None and fallback_path:
        display_name = fallback_path.split(".")[-1]

    return display_name


@dataclass(frozen=True)
class SimulatedResponse:
    """Single response item for LLM simulation.

    Encapsulates response content as ordered blocks. Passes simulation
    data between orchestrator and HTTP interceptor layers.

    Args:
        content: Exception to raise, or empty string for normal responses.
        blocks: Ordered blocks (str and ToolCall objects). Always present
            for normal responses; adapters use this for response building.
        raw_payload: Raw provider JSON passthrough (for RawLLMResponse).
            When set, adapters use this directly instead of building from blocks.
    """

    content: str | Exception
    blocks: list[Any] | None = None
    raw_payload: dict[str, Any] | None = None


def _normalize_to_list(
    response: str | None,
    responses: ResponsesInput,
) -> list[Any]:
    """Normalize all input forms to a single list.

    Args:
        response: Deprecated single response param.
        responses: Single or list of responses.

    Returns:
        List of response items (copies list to avoid mutation).
    """
    if response is not None:
        return [response]
    if responses is None:
        return []
    if isinstance(responses, (str, Exception, ToolCall, LLMResponse, RawLLMResponse, dict)):
        return [responses]
    return list(responses)


class SimulationOrchestrator:
    """Orchestrates simulation rules and patching for Construct.

    Manages the lifecycle of simulations including:
    - Storing simulation rules
    - Applying/restoring patches
    - Tracking simulation state
    - Coordinating with HTTP interceptor for LLM simulations
    """

    def __init__(
        self,
        lifecycle: LifecycleManager,
        http_interceptor: HttpInterceptor,
    ) -> None:
        """Initialize orchestrator with dependencies.

        Args:
            lifecycle: Lifecycle manager for span tracking.
            http_interceptor: HTTP interceptor for LLM simulations.
        """
        self._lifecycle = lifecycle
        self._http_interceptor = http_interceptor
        self._simulations: dict[str, SimulationRule] = {}
        self._originals: dict[str, Any] = {}
        self._active: bool = False
        self._http_simulation_enabled: bool = False
        self._simulation_tracker = SimulationTracker()
        self._tool_tracker = ToolAgentTracker()
        self._agent_tracker = ToolAgentTracker()
        # HTTP intercepts at adapter level; spans show provider_id, not adapter
        self._adapter_to_provider: dict[str, str] = {}
        # Functions with __code__ swapped for capture-safe simulation
        self._registered_trampolines: list[Callable[..., Any]] = []

    @property
    def simulation_tracker(self) -> SimulationTracker:
        """Get the LLM simulation tracker for validation."""
        return self._simulation_tracker

    @property
    def tool_tracker(self) -> ToolAgentTracker:
        """Get the tool simulation tracker for validation."""
        return self._tool_tracker

    @property
    def agent_tracker(self) -> ToolAgentTracker:
        """Get the agent simulation tracker for validation."""
        return self._agent_tracker

    @property
    def http_simulation_enabled(self) -> bool:
        """Check if HTTP simulation is enabled."""
        return self._http_simulation_enabled

    def get_rule(self, canonical_key: str) -> SimulationRule | None:
        """Get the simulation rule for a target.

        Args:
            canonical_key: The target's fully-qualified dotted path.

        Returns:
            The simulation rule if registered, None otherwise.
        """
        return self._simulations.get(canonical_key)

    def activate(self) -> None:
        """Activate simulations and apply all pending patches."""
        self._active = True
        self._http_interceptor.start()
        # Linked targets use dispatch, skip setattr patching
        for target_path, rule in self._simulations.items():
            if not getattr(rule, "is_linked", False):
                self._apply_patch(target_path)

    def deactivate(self) -> None:
        """Deactivate simulations and restore originals."""
        self._active = False
        self._stop_http_interceptor()
        self._stop_patches()
        self._restore_originals()
        self._restore_trampolines()
        self._clear_state()

    def handle_http_call(
        self,
        adapter: str,
        messages: list[dict[str, Any]],
        model: str | None,
        response_text: str,
        agent: str | None,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        """Handle HTTP interception callback by creating LLMCall span.

        Always creates a new LLMCall. LLMScope (from @link_llm decorator) is
        transparent for parent attribution - the LLMCall references it via
        llm_scope_id for grouping but gets its parent_span_id from structural
        spans (Agent, Tool).

        Args:
            adapter: HTTP adapter name from interceptor (e.g., "openai").
            messages: List of message dicts from the request body.
            model: Model identifier from the request.
            response_text: The simulated response text.
            agent: Name of the agent that made this call, or None.
            tool_calls: List of tool call dicts the LLM emitted in its response.
        """
        # Translate adapter to provider_id for tracking
        # If no mapping exists, adapter IS the provider (built-in case)
        provider_id = self._adapter_to_provider.get(adapter, adapter)
        self._simulation_tracker.mark_triggered(provider_id)

        span = self._create_llm_span(
            provider=provider_id,
            messages=messages,
            model=model,
            response=response_text,
            agent_name=agent,
            tool_calls=tool_calls if tool_calls else None,
        )
        span.simulated = True  # Mark as simulated for verification tracking
        with self._lifecycle.start_span(span):
            pass  # Span lifecycle managed by context manager

    def simulate_tool(
        self,
        target: str | Callable[..., Any],
        result: Any = None,
        results: list[Any] | None = None,
        side_effect: Callable[..., Any] | None = None,
        optional: bool = False,
    ) -> None:
        """Simulate a tool with lifecycle recording.

        Args:
            target: Dotted-path string to a function/method or function/method object.
            result: Single static response value for the tool.
            results: List of response values for sequential calls.
            side_effect: Callable for dynamic behavior.
            optional: If True, won't raise if unused.

        Raises:
            ValueError: If multiple response parameters are provided, if target
                is not patchable, or if string path resolves to a class.
        """
        validate_simulation_params(result, results, side_effect)
        validate_target_is_patchable(target, SimulationType.TOOL)

        if isinstance(target, str):
            validate_string_path(target)
            _validate_string_path_target(
                target, "tool", "Decorate with @link_tool or use tenro.register()."
            )

        resolved_display_name = _resolve_display_name(target, "")
        all_paths = resolve_all_target_paths(target, is_tool=True)
        if not all_paths:
            # Fallback for targets that can't be resolved
            all_paths = validate_and_resolve_target(target, SimulationType.TOOL)

        if not resolved_display_name and all_paths:
            resolved_display_name = all_paths[0].split(".")[-1]

        # Detect if target is linked as tool (dispatch handles simulation)
        # Note: Only @link_tool is valid, not @link_agent or @link_llm
        # Resolve string targets first to check the actual callable
        resolved_target = _resolve_string_target(target)
        is_tool_linked = _is_linked_type(resolved_target, "tool")

        # For capture-safe simulation: need actual function object, not string/identity.
        # Check identity-stamped originals without dispatch wrapper.
        # Use resolved_target for registration check (handles string paths)
        is_registered_function = False
        has_identity_only = hasattr(resolved_target, "_tenro_identity_key") and not hasattr(
            resolved_target, "_tenro_linked_type"
        )
        needs_registration_check = (
            callable(resolved_target)
            and not inspect.isclass(resolved_target)
            and (not is_tool_linked or has_identity_only)
        )
        if needs_registration_check:
            from tenro.simulate._register import is_registered

            is_registered_function = is_registered(resolved_target)

        # Fail fast for non-linked, non-registered module-level functions.
        if _is_module_level_function(target) and not is_tool_linked and not is_registered_function:
            target_name = getattr(target, "__qualname__", repr(target))
            _raise_non_linked_error(
                target_name,
                "tool",
                "Decorate with @link_tool or use tenro.register().",
                failure_reason=(
                    "Module-level function is not linked. "
                    "Captured references bypass setattr patching."
                ),
                target_type=_detect_target_type(target),
            )

        # Detect if target is a generator or async generator function
        is_generator = _is_generator_target(target)

        # Trampoline enables capture-safe simulation for registered functions
        if is_registered_function and callable(resolved_target):
            from tenro.simulate._register import install_trampoline

            canonical_key = all_paths[0] if all_paths else repr(resolved_target)
            install_trampoline(resolved_target, canonical_key)
            self._registered_trampolines.append(resolved_target)

        # Install trampoline on registered __wrapped__ (wrapper→original sync)
        if is_tool_linked and hasattr(target, "__wrapped__"):
            from tenro.simulate._register import install_trampoline, is_registered

            wrapped = target.__wrapped__
            if callable(wrapped) and is_registered(wrapped):
                canonical_key = all_paths[0] if all_paths else repr(wrapped)
                install_trampoline(wrapped, canonical_key)
                self._registered_trampolines.append(wrapped)

        # Both linked targets and registered functions use dispatch
        uses_dispatch = is_tool_linked or is_registered_function

        for path in all_paths:
            self._tool_tracker.register(path, optional=optional)
            path_result_sequence = normalize_result_sequence(result, results)

            # Only dispatch-based targets supported (no setattr patching)
            if not uses_dispatch:
                target_name = path.split(".")[-1] if path else repr(target)
                _raise_non_linked_error(
                    target_name,
                    "tool",
                    "Decorate with @link_tool or use tenro.register().",
                    failure_reason=(
                        "Target is not linked or registered. "
                        "Setattr patching is not supported at simulate() time."
                    ),
                    target_type=_detect_target_type(target),
                )

            # Dispatch-based targets (linked or registered) - store raw simulation params
            # For generators, store results as generator_items (yield all at once)
            # For non-generators:
            #   - result= (single) -> returns_value (infinite)
            #   - results= (list) -> result_sequence (consumed per call)
            if is_generator:
                rule = SimulationRule(
                    side_effect=side_effect,
                    generator_items=path_result_sequence,
                    is_linked=True,
                    is_generator=True,
                    strategy="dispatch",
                )
            elif result is not None:
                # Single result - infinite supply via returns_value
                rule = SimulationRule(
                    side_effect=side_effect,
                    returns_value=result,
                    is_linked=True,
                    strategy="dispatch",
                )
            else:
                # results= list - consumed per call via result_sequence
                rule = SimulationRule(
                    side_effect=side_effect,
                    result_sequence=path_result_sequence,
                    is_linked=True,
                    strategy="dispatch",
                )

            self._simulations[path] = rule

    def simulate_agent(
        self,
        target: str | Callable[..., Any],
        result: Any = None,
        results: list[Any] | None = None,
        side_effect: Callable[..., Any] | None = None,
        optional: bool = False,
    ) -> None:
        """Simulate an agent with lifecycle recording.

        Args:
            target: Dotted-path string to a function/method or function/method object.
            result: Single static response value for the agent.
            results: List of response values for sequential calls.
            side_effect: Callable for dynamic behavior.
            optional: If True, won't raise if unused.

        Raises:
            ValueError: If multiple response parameters are provided, if target
                is not patchable, or if string path resolves to a class.
        """
        validate_simulation_params(result, results, side_effect)
        validate_target_is_patchable(target, SimulationType.AGENT)

        if isinstance(target, str):
            validate_string_path(target)
            _validate_string_path_target(target, "agent", "Decorate with @link_agent.")

        resolved_display_name = _resolve_display_name(target, "")
        all_paths = resolve_all_target_paths(target)
        if not all_paths:
            all_paths = validate_and_resolve_target(target, SimulationType.AGENT)

        if not resolved_display_name and all_paths:
            resolved_display_name = all_paths[0].split(".")[-1]

        # Detect if target is linked as agent (dispatch handles simulation)
        # Note: Only @link_agent is valid, not @link_tool or @link_llm
        # Resolve string targets first to check the actual callable
        resolved_target = _resolve_string_target(target)
        is_agent_linked = _is_linked_type(resolved_target, "agent")

        # Detect if target is a generator or async generator function
        is_generator = _is_generator_target(target)

        # Fail fast for non-linked module-level functions.
        if _is_module_level_function(target) and not is_agent_linked:
            target_name = getattr(target, "__qualname__", repr(target))
            _raise_non_linked_error(
                target_name,
                "agent",
                "Decorate with @link_agent.",
                failure_reason=(
                    "Module-level function is not linked. "
                    "Captured references bypass setattr patching."
                ),
                target_type=_detect_target_type(target),
            )

        for path in all_paths:
            self._agent_tracker.register(path, optional=optional)
            path_result_sequence = normalize_result_sequence(result, results)

            # Only linked targets supported (no setattr patching)
            if not is_agent_linked:
                target_name = path.split(".")[-1] if path else repr(target)
                _raise_non_linked_error(
                    target_name,
                    "agent",
                    "Decorate with @link_agent.",
                    failure_reason=(
                        "Target is not linked. "
                        "Setattr patching is not supported at simulate() time."
                    ),
                    target_type=_detect_target_type(target),
                )

            # Linked function targets use dispatch - store raw simulation params
            # For generators, store results as generator_items (yield all at once)
            # For non-generators:
            #   - result= (single) -> returns_value (infinite)
            #   - results= (list) -> result_sequence (consumed per call)
            if is_generator:
                rule = SimulationRule(
                    side_effect=side_effect,
                    generator_items=path_result_sequence,
                    is_linked=True,
                    is_generator=True,
                    strategy="dispatch",
                )
            elif result is not None:
                # Single result - infinite supply via returns_value
                rule = SimulationRule(
                    side_effect=side_effect,
                    returns_value=result,
                    is_linked=True,
                    strategy="dispatch",
                )
            else:
                # results= list - consumed per call via result_sequence
                rule = SimulationRule(
                    side_effect=side_effect,
                    result_sequence=path_result_sequence,
                    is_linked=True,
                    strategy="dispatch",
                )

            self._simulations[path] = rule

    def simulate_llm(
        self,
        target: str | Callable[..., Any] | None = None,
        provider: str | None = None,
        adapter: str | None = None,
        *,
        response: str | None = None,
        responses: ResponsesInput = None,
        model: str | None = None,
        tool_calls: list[SimToolCall | str | dict[str, Any]] | None = None,
        use_http: bool | None = None,
        optional: bool = False,
        **response_kwargs: Any,
    ) -> None:
        """Simulate LLM calls with provider detection and lifecycle tracking.

        Args:
            target: Optional dotted-path string to a function/method to simulate.
            provider: LLM provider ID for tracking/verification.
            adapter: HTTP adapter name (openai, anthropic, gemini) for endpoint
                selection. If None, defaults to provider value.
            response: Single string response.
            responses: Response(s) for sequential calls. Each item can be:
                - str: Plain text response
                - Exception: Raise on that call
                - ToolCall: Single tool call (shorthand)
                - list[ToolCall]: Multiple tool calls per turn
                - list[str | ToolCall]: Ordered blocks with interleaved text
                - LLMResponse: Explicit block structure
                - dict: Legacy format {"text": ..., "tool_calls": [...]} (deprecated)
            model: Model identifier override.
            tool_calls: DEPRECATED. Tool calls to attach to first response.
                Use ToolCall in responses instead: responses=[ToolCall(...)]
            use_http: Force HTTP interception or dispatch mode.
            optional: If True, won't raise if unused.
            **response_kwargs: Provider-specific options.

        Raises:
            TenroValidationError: If response parameters are invalid or conflicting.

        Examples:
            Simulate a simple text response::

                construct.simulate_llm(Provider.OPENAI, response="Hello!")

            Simulate tool calls with shorthand (preferred)::

                construct.simulate_llm(
                    Provider.OPENAI,
                    responses=[ToolCall(search, query="AI"), "Found results!"],
                )

            Multiple tool calls per turn::

                construct.simulate_llm(
                    Provider.OPENAI,
                    responses=[[ToolCall(search, q="A"), ToolCall(fetch, id=1)]],
                )

            Interleaved text and tool calls::

                construct.simulate_llm(
                    Provider.OPENAI,
                    responses=[["Let me search", ToolCall(search, q="AI"), "Done!"]],
                )
        """
        from tenro._construct.http.interceptor import (
            get_provider_endpoints,
            get_supported_providers,
        )

        if tool_calls is not None:
            warnings.warn(
                "The 'tool_calls=' parameter is deprecated. "
                "Use ToolCall objects directly in responses=: "
                "responses=[ToolCall('search', query='AI')]",
                DeprecationWarning,
                stacklevel=3,
            )

        if responses is not None and isinstance(responses, list):
            for resp in responses:
                if isinstance(resp, dict):
                    warnings.warn(
                        "Dict format in responses is deprecated. "
                        "Use LLMResponse or ToolCall: "
                        "responses=[LLMResponse(blocks=['text', ToolCall(...)])]",
                        DeprecationWarning,
                        stacklevel=3,
                    )
                    break

        # Validate tool_calls= conflicts with embedded tool calls
        if tool_calls is not None:
            # Consider both deprecated response= and responses= parameters
            raw_items_for_check: list[Any] = []
            if response is not None:
                raw_items_for_check = [response]
            elif responses is not None:
                raw_items_for_check = (
                    list(responses) if isinstance(responses, list) else [responses]
                )

            # Empty responses with tool_calls= is invalid
            if not raw_items_for_check:
                from tenro.errors import TenroValidationError

                raise TenroValidationError(
                    "The 'tool_calls=' parameter requires at least one response to apply to. "
                    "Provide responses=['text'] or use responses=[ToolCall(...)] directly."
                )

            # Check each response item for conflicts
            for item in raw_items_for_check:
                # RawLLMResponse can't have tool calls injected
                if isinstance(item, RawLLMResponse):
                    from tenro.errors import TenroValidationError

                    raise TenroValidationError(
                        "Cannot use 'tool_calls=' with RawLLMResponse in responses. "
                        "RawLLMResponse is a passthrough format that cannot be modified. "
                        "Use responses=[ToolCall(...)] or LLMResponse(blocks=[...]) instead."
                    )

                # Check if item contributes tool_calls (conflict)
                if contributes_tool_calls(item):
                    from tenro.errors import TenroValidationError

                    raise TenroValidationError(
                        "Cannot use 'tool_calls=' when responses contains ToolCall items. "
                        "Use the shorthand directly: responses=[ToolCall('search', query='AI')] "
                        "or responses=[LLMResponse(blocks=['text', ToolCall(...)])]"
                    )

        # Validate mutual exclusivity (only checks if both provided)
        validate_llm_simulation_params(response, "present" if responses is not None else None)

        raw_items = _normalize_to_list(response, responses)
        normalized_tool_calls = normalize_tool_calls(tool_calls)

        simulated_responses: list[SimulatedResponse] = []
        has_embedded_tool_calls = False

        for item in raw_items:
            if contributes_tool_calls(item):
                has_embedded_tool_calls = True

            parsed = parse_response_item(item)
            if isinstance(parsed, Exception):
                simulated_responses.append(SimulatedResponse(content=parsed))
            elif isinstance(parsed, RawLLMResponse):
                simulated_responses.append(
                    SimulatedResponse(content="", raw_payload=parsed.payload)
                )
            elif isinstance(parsed, LLMResponse):
                # Always use blocks - parser normalizes all inputs to LLMResponse(blocks=[...])
                simulated_responses.append(SimulatedResponse(content="", blocks=parsed.blocks))

        if normalized_tool_calls and simulated_responses:
            first_is_exception = isinstance(simulated_responses[0].content, Exception)
            if first_is_exception:
                pass
            elif not has_embedded_tool_calls:
                # Merge tool_calls into first response's blocks
                merged_blocks = list(simulated_responses[0].blocks or []) + list(
                    normalized_tool_calls
                )
                simulated_responses[0] = SimulatedResponse(content="", blocks=merged_blocks)

        # Fail-fast for non-simulatable callable targets when use_http=False.
        # This must happen BEFORE provider detection which may fail on plain functions.
        # Resolve string paths to check the actual callable.
        resolved_target = _resolve_string_target(target) if target else None
        if (
            resolved_target is not None
            and callable(resolved_target)
            and use_http is False  # Explicit setattr mode requested
        ):
            is_simulatable = _is_linked_target(resolved_target) or _is_registered_target(
                resolved_target
            )
            if not is_simulatable:
                from tenro.errors.simulation import (
                    SimulationDiagnostic,
                    TenroSimulationSetupError,
                )

                target_name = getattr(resolved_target, "__qualname__", repr(target))
                diagnostic = SimulationDiagnostic(
                    target_path=target_name,
                    target_type=_detect_target_type(resolved_target),
                    is_linked=False,
                    failure_reason=(
                        "Target is not linked or registered; capture-safe interception unavailable."
                    ),
                    recommended_fix=(
                        "Decorate with @link_llm, use tenro.register(), "
                        "or use HTTP mode (use_http=True)."
                    ),
                )
                msg = (
                    f"Cannot simulate non-linked LLM target '{target_name}'.\n\n"
                    f"Reason: {diagnostic.failure_reason}\n"
                    f"Fix: {diagnostic.recommended_fix}"
                )
                raise TenroSimulationSetupError(msg, diagnostic=diagnostic)

        effective_adapter = adapter if adapter is not None else provider
        if effective_adapter is None:
            effective_adapter = resolve_provider_from_target(target, None, self._detect_provider)

        custom_target_provided = target is not None
        supported_providers = get_supported_providers()
        use_http_resolved = should_use_http_interception(
            use_http, custom_target_provided, effective_adapter, supported_providers
        )

        provider_endpoints = get_provider_endpoints()
        if use_http_resolved and effective_adapter in provider_endpoints:
            self._simulate_llm_http(
                provider=provider or effective_adapter,
                adapter=effective_adapter,
                simulated_responses=simulated_responses,
                model=model,
                optional=optional,
                **response_kwargs,
            )
        else:
            self._simulate_llm_dispatch(
                target=target,
                provider=provider or effective_adapter,
                adapter=effective_adapter,
                simulated_responses=simulated_responses,
                model=model,
                optional=optional,
                **response_kwargs,
            )

    def _detect_provider(self, target: str) -> str:
        """Auto-detect provider from target path."""
        return ProviderSchemaFactory.detect_provider(target)

    def _create_provider_response(
        self, provider: str, content: str, **kwargs: Any
    ) -> ProviderResponse:
        """Create a provider-specific response object."""
        return ProviderSchemaFactory.create_response(provider, content, **kwargs)

    def _create_llm_span(
        self,
        provider: str,
        messages: list[dict[str, Any]],
        model: str | None = None,
        response: str | None = None,
        agent_name: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> LLMCall:
        """Create LLMCall with context linking.

        Centralizes LLMCall creation to ensure consistent llm_scope_id
        and caller info propagation from @link_llm decorator.

        Args:
            provider: LLM provider name.
            messages: Request messages.
            model: Model identifier.
            response: Response text.
            agent_name: Agent that made this call.
            tool_calls: Tool calls emitted by the LLM in its response.

        Returns:
            LLMCall span with context linked.
        """
        llm_scope = context.get_nearest_llm_scope()
        return LLMCall(
            id=str(uuid7()),
            trace_id=str(uuid7()),
            start_time=time.time(),
            provider=provider,
            model=model,
            messages=messages,
            response=response,
            agent_name=agent_name,
            tool_calls=tool_calls,
            llm_scope_id=llm_scope.id if llm_scope else None,
            caller_signature=llm_scope.caller_signature if llm_scope else None,
            caller_location=llm_scope.caller_location if llm_scope else None,
        )

    def _simulate_llm_http(
        self,
        provider: str,
        adapter: str,
        simulated_responses: list[SimulatedResponse],
        model: str | None = None,
        optional: bool = False,
        **response_kwargs: Any,
    ) -> None:
        """Simulate LLM using HTTP interception.

        Args:
            provider: Provider ID for tracking/verification (e.g., "mistral").
            adapter: HTTP adapter for endpoint selection (e.g., "openai").
            simulated_responses: List of response items with content and optional
                tool_calls.
            model: Model identifier override.
            optional: If True, won't raise if unused.
            **response_kwargs: Provider-specific options.
        """
        if not simulated_responses:
            raise ValueError("HTTP simulation requires at least one response")

        if model is not None:
            response_kwargs["model"] = model

        for i, resp in enumerate(simulated_responses):
            if isinstance(resp.content, Exception) and resp.blocks:
                raise ValueError(
                    f"Response at index {i} is an Exception, "
                    "but blocks were specified. "
                    "Exceptions cannot have associated blocks."
                )

        self._http_simulation_enabled = True
        self._simulation_tracker.register(provider, optional=optional)
        self._adapter_to_provider[adapter] = provider

        # Determine HTTP routing: use provider's endpoint if globally registered,
        # otherwise fall back to adapter's endpoint (for instance-registered providers)
        from tenro._construct.http.interceptor import get_provider_endpoints

        endpoints = get_provider_endpoints()
        http_provider = provider if provider in endpoints else adapter

        self._http_interceptor.simulate_provider(
            http_provider,
            simulated_responses,
            adapter=adapter,  # Pass adapter for response format
            **response_kwargs,
        )
        # Note: No need to start() here - the interceptor is already started in activate().
        # The unified handler in the interceptor checks _response_queue for simulations.

    def _simulate_llm_dispatch(
        self,
        target: str | Callable[..., Any] | None,
        provider: str,
        adapter: str,
        simulated_responses: list[SimulatedResponse],
        model: str | None = None,
        optional: bool = False,
        **response_kwargs: Any,
    ) -> None:
        """Simulate LLM via dispatch for @link_llm or registered targets.

        Non-linked callable targets raise TenroSimulationSetupError to prevent silent
        failures when captured references bypass setattr patching.

        Args:
            target: Target function/method to simulate or None for provider default.
            provider: LLM provider ID for tracking/verification.
            adapter: HTTP adapter name for response schema selection.
            simulated_responses: List of responses to return sequentially.
            model: Model identifier override.
            optional: If True, won't raise if unused.
            **response_kwargs: Provider-specific response options.
        """
        if target is None:
            target = ProviderSchemaFactory.get_default_target(provider)
            if target is None:
                available = list(ProviderSchemaFactory._default_targets.keys())
                raise ValueError(
                    f"Provider '{provider}' has no default target.\n"
                    f"Available providers with defaults: {', '.join(available)}"
                )

        validate_target_is_patchable(target, SimulationType.LLM)
        if isinstance(target, str):
            validate_string_path(target)

        # Resolve string paths to actual callable for checking
        resolved_target = _resolve_string_target(target) if isinstance(target, str) else target

        # Only linked or registered functions support capture-safe dispatch.
        # Recognizes identity-stamped originals (from __wrapped__).
        if resolved_target is not None and callable(resolved_target):
            is_simulatable = _is_linked_target(resolved_target) or _is_registered_target(
                resolved_target
            )
            if not is_simulatable:
                from tenro.errors.simulation import (
                    SimulationDiagnostic,
                    TenroSimulationSetupError,
                )

                target_name = getattr(resolved_target, "__qualname__", repr(target))
                diagnostic = SimulationDiagnostic(
                    target_path=target_name,
                    target_type=_detect_target_type(resolved_target),
                    is_linked=False,
                    failure_reason=(
                        "Target is not linked or registered; capture-safe interception unavailable."
                    ),
                    recommended_fix=(
                        "Decorate with @link_llm, use tenro.register(), "
                        "or use HTTP mode (use_http=True)."
                    ),
                )
                msg = (
                    f"Cannot simulate non-linked LLM target '{target_name}' "
                    f"with use_http=False.\n\n"
                    f"Reason: {diagnostic.failure_reason}\n"
                    f"Fix: {diagnostic.recommended_fix}"
                )
                raise TenroSimulationSetupError(msg, diagnostic=diagnostic)

        self._simulation_tracker.register(provider, optional=optional)

        if callable(target) and not isinstance(target, str):
            try:
                llm_paths = validate_and_resolve_target(target, SimulationType.LLM)
                llm_path = llm_paths[0] if llm_paths else str(target)
            except ValueError:
                qualname = getattr(target, "__qualname__", "")
                module = getattr(target, "__module__", "")
                llm_path = f"{module}.{qualname}" if module and qualname else str(target)
        else:
            llm_path = str(target)

        if model is not None:
            response_kwargs["model"] = model

        response_index = {"current": 0}

        def _dispatch_side_effect(*args: Any, **kwargs: Any) -> str:
            """Side effect for dispatch-based simulation (returns content string).

            For @link_llm decorated functions, dispatch handles both captured refs
            and module-level calls since the decorator wrapper is always invoked.
            """
            self._simulation_tracker.mark_triggered(provider)
            idx = response_index["current"]
            response_index["current"] += 1

            if idx >= len(simulated_responses):
                raise IndexError(
                    f"Simulation exhausted: {llm_path} called {idx + 1} times "
                    f"but only {len(simulated_responses)} responses provided"
                )

            sim_resp = simulated_responses[idx]

            # Handle exceptions stored in content
            if isinstance(sim_resp.content, Exception):
                raise sim_resp.content

            # Extract text from blocks for dispatch mode return value
            blocks = sim_resp.blocks or []
            text_parts = [b for b in blocks if isinstance(b, str)]
            return "".join(text_parts)

        # Only linked/registered targets supported (via dispatch, not setattr)
        # Use resolved_target from earlier (handles string paths)
        is_simulatable = (
            resolved_target is not None
            and callable(resolved_target)
            and (_is_linked_target(resolved_target) or _is_registered_target(resolved_target))
        )

        # Only linked/registered targets supported (no setattr patching)
        if not is_simulatable:
            from tenro.errors.simulation import (
                SimulationDiagnostic,
                TenroSimulationSetupError,
            )

            target_name = llm_path.split(".")[-1] if llm_path else repr(target)
            diagnostic = SimulationDiagnostic(
                target_path=target_name,
                target_type=_detect_target_type(resolved_target)
                if resolved_target
                else "string_path",
                is_linked=False,
                failure_reason=(
                    "Target is not linked or registered; capture-safe interception unavailable."
                ),
                recommended_fix=(
                    "Decorate with @link_llm, use tenro.register(), "
                    "or use HTTP mode (use_http=True)."
                ),
            )
            msg = (
                f"Cannot simulate non-linked LLM target '{target_name}'.\n\n"
                f"Reason: {diagnostic.failure_reason}\n"
                f"Fix: {diagnostic.recommended_fix}"
            )
            raise TenroSimulationSetupError(msg, diagnostic=diagnostic)

        # Install trampoline for registered functions (capture-safe)
        is_registered = _is_registered_target(resolved_target) if resolved_target else False
        if is_registered and callable(resolved_target):
            from tenro.simulate._register import install_trampoline

            install_trampoline(resolved_target, llm_path)
            self._registered_trampolines.append(resolved_target)

        # For linked/registered functions, use dispatch (returns string)
        is_generator = _is_generator_target(resolved_target or target)

        if is_generator:
            # For generators, use generator_items to yield all responses
            # Extract text from blocks (same as _dispatch_side_effect)
            generator_items: list[object] = []
            for resp in simulated_responses:
                if isinstance(resp.content, Exception):
                    generator_items.append(resp.content)
                else:
                    blocks = resp.blocks or []
                    text_parts = [b for b in blocks if isinstance(b, str)]
                    generator_items.append("".join(text_parts))
            dispatch_rule = SimulationRule(
                side_effect=_dispatch_side_effect,
                generator_items=generator_items,
                is_linked=True,
                is_generator=True,
                strategy="dispatch",
                llm_provider=provider,
                llm_model=model,
            )
        else:
            dispatch_rule = SimulationRule(
                side_effect=_dispatch_side_effect,
                strategy="dispatch",
                llm_provider=provider,
                llm_model=model,
            )
        self._simulations[llm_path] = dispatch_rule

    def _apply_patch(self, target_path: str) -> None:
        """Apply a single patch for a simulation target."""
        if target_path in self._originals:
            return

        container, attr_name = parse_dotted_path(target_path)

        if not hasattr(container, attr_name):
            msg = f"'{container}' has no attribute '{attr_name}'"
            raise AttributeError(msg)

        original = getattr(container, attr_name)
        self._originals[target_path] = original

        rule = self._simulations[target_path]

        def make_wrapper(sim_rule: SimulationRule) -> Callable[..., Any]:
            is_async = inspect.iscoroutinefunction(inspect.unwrap(original))

            def _execute_side_effect(side_effect: Any, *args: Any, **kwargs: Any) -> Any:
                if callable(side_effect):
                    return side_effect(*args, **kwargs)
                elif isinstance(side_effect, list):
                    if side_effect:
                        return side_effect[0](*args, **kwargs)
                    return None
                elif isinstance(side_effect, BaseException):
                    raise side_effect
                elif isinstance(side_effect, type) and issubclass(side_effect, BaseException):
                    raise side_effect()
                else:
                    raise TypeError("side_effect must be callable, list, or exception")

            if is_async:

                @functools.wraps(original)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    if sim_rule.side_effect:
                        result = _execute_side_effect(sim_rule.side_effect, *args, **kwargs)
                        if asyncio.iscoroutine(result):
                            return await result
                        return result
                    return sim_rule.returns_value

                wrapper: Callable[..., Any] = async_wrapper
            else:

                @functools.wraps(original)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    if sim_rule.side_effect:
                        result = _execute_side_effect(sim_rule.side_effect, *args, **kwargs)
                        if asyncio.iscoroutine(result):
                            try:
                                loop = asyncio.get_running_loop()
                            except RuntimeError:
                                loop = None
                            if loop is not None:
                                # Can't nest asyncio.run() in running loop
                                import concurrent.futures

                                with concurrent.futures.ThreadPoolExecutor() as pool:
                                    future = pool.submit(asyncio.run, result)
                                    return future.result()
                            else:
                                return asyncio.run(result)
                        return result
                    return sim_rule.returns_value

                wrapper = sync_wrapper

            if hasattr(original, "_tenro_full_path"):
                wrapper._tenro_full_path = original._tenro_full_path  # type: ignore[attr-defined]
            if hasattr(original, "_tenro_linked_name"):
                wrapper._tenro_linked_name = original._tenro_linked_name  # type: ignore[attr-defined]
            if hasattr(original, "_tenro_linked_type"):
                wrapper._tenro_linked_type = original._tenro_linked_type  # type: ignore[attr-defined]
            if hasattr(original, "_tenro_target_paths"):
                wrapper._tenro_target_paths = original._tenro_target_paths  # type: ignore[attr-defined]
            if hasattr(original, "_tenro_display_name"):
                wrapper._tenro_display_name = original._tenro_display_name  # type: ignore[attr-defined]

            return wrapper

        setattr(container, attr_name, make_wrapper(rule))

    def _stop_http_interceptor(self) -> None:
        """Stop HTTP interception."""
        # Always stop since it always starts in activate()
        self._http_interceptor.stop()
        self._http_simulation_enabled = False

    def _stop_patches(self) -> None:
        """Stop active patches.

        Patches are restored in `_restore_originals`, not here.
        """
        pass

    def _restore_originals(self) -> None:
        """Restore patched functions."""
        for tool_name, original in self._originals.items():
            container, attr_name = parse_dotted_path(tool_name)
            setattr(container, attr_name, original)

    def _restore_trampolines(self) -> None:
        """Restore __code__ for registered functions after simulation ends."""
        from tenro.simulate._register import uninstall_trampoline

        for func in self._registered_trampolines:
            uninstall_trampoline(func)
        self._registered_trampolines.clear()

    def _clear_state(self) -> None:
        """Clear patching state while preserving trace data."""
        self._originals.clear()
        self._adapter_to_provider.clear()


__all__ = ["SimulationOrchestrator"]
