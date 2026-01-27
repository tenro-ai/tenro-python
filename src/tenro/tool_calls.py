# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Tool call types and helpers for LLM simulation."""

from __future__ import annotations

import dataclasses
import datetime
import functools
import inspect
import json
import warnings
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from tenro.errors.warnings import TenroCoercionWarning, warn

JSONPrimitive = str | int | float | bool | None
JSONValue = JSONPrimitive | dict[str, "JSONValue"] | list["JSONValue"]
JSONObject = dict[str, JSONValue]


@dataclass(frozen=True, init=False)
class ToolCall:
    """Provider-agnostic tool call representation.

    Can be created in three ways:
    - ToolCall(callable, **kwargs)  # preferred: type-safe
    - ToolCall("name", **kwargs)    # string form
    - ToolCall(name="...", arguments={...})  # explicit form

    Args:
        tool_or_name: Tool callable or name string (positional).
        name: Tool name (keyword, for explicit form).
        arguments: Tool arguments as dict (keyword, for explicit form).
        call_id: Correlation ID for matching calls to results. Auto-generated if None.
        _name: Override the tool name (useful when __name__ collides).

    Examples:
        >>> ToolCall(search, query="AI")        # callable + kwargs
        ToolCall(name='search', arguments={'query': 'AI'}, call_id='call_...')

        >>> ToolCall("search", query="AI")      # string + kwargs
        ToolCall(name='search', arguments={'query': 'AI'}, call_id='call_...')

        >>> ToolCall(name="search", arguments={"query": "AI"})  # explicit
        ToolCall(name='search', arguments={'query': 'AI'}, call_id='call_...')

        >>> ToolCall(name="get_time")  # Zero arguments
        ToolCall(name='get_time', arguments={}, call_id='call_...')
    """

    name: str
    arguments: JSONObject
    call_id: str

    def __init__(
        self,
        tool_or_name: str | Any | None = None,
        *,
        name: str | None = None,
        arguments: JSONObject | None = None,
        call_id: str | None = None,
        _name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Smart constructor supporting callable, string, or explicit form."""
        # Guard against ambiguous combinations
        if tool_or_name is not None and name is not None:
            raise TypeError(
                "Cannot use both positional tool/name and name= keyword. "
                "Use ToolCall('name', ...) OR ToolCall(name='name', ...)."
            )

        # kwargs merge requires dict; catch early with clear message
        if arguments is not None and not isinstance(arguments, dict):
            raise TypeError(
                f"arguments= must be a dict, got {type(arguments).__name__}. "
                "Use ToolCall('name', **kwargs) or ToolCall(name='...', arguments={{...}})."
            )

        resolved_name = _resolve_tool_name(tool_or_name, _name, name)
        resolved_args = {**(arguments or {}), **kwargs}
        coerced_args = _validate_json_serializable(resolved_args)

        if call_id is not None:
            if not isinstance(call_id, str):
                raise TypeError(f"call_id must be str, got {type(call_id).__name__}")
            if call_id == "":
                raise TypeError("call_id must be non-empty str (got '')")
            resolved_call_id = call_id
        else:
            resolved_call_id = f"call_{uuid4().hex[:12]}"

        # Name must be str for LLM wire format
        if not isinstance(resolved_name, str):
            raise TypeError(f"name must be str, got {type(resolved_name).__name__}")
        if resolved_name == "":
            raise TypeError("name must be non-empty str")

        # frozen=True prevents self.attr = val; object.__setattr__ is the
        # standard pattern for custom __init__ logic in frozen dataclasses.
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "arguments", coerced_args)
        object.__setattr__(self, "call_id", resolved_call_id)


def tc(
    tool: str | Any,
    *,
    _name: str | None = None,
    call_id: str | None = None,
    **arguments: Any,
) -> ToolCall:
    """Create a ToolCall from a tool name or callable.

    .. deprecated::
        Use ``ToolCall()`` directly instead.

    Args:
        tool: Tool name as string, or callable reference.
        _name: Override the tool name (useful when __name__ collides).
        call_id: Explicit call ID (auto-generated if None).
        **arguments: Tool arguments as keyword args (coerced to JSON-serializable).

    Returns:
        ToolCall instance.

    Raises:
        TypeError: If tool is not str/callable, call_id is empty, or
            arguments not JSON-serializable.

    Note:
        If your tool has an arg named `call_id`, pass it via
        ``ToolCall(..., arguments={"call_id": ...})``.

    Examples:
        >>> tc("search", query="AI")
        ToolCall(name="search", arguments={"query": "AI"}, ...)

        >>> tc(get_weather, city="Paris")
        ToolCall(name="get_weather", arguments={"city": "Paris"}, ...)

        >>> tc(my_search, _name="search_docs", query="AI")  # Override name
        ToolCall(name="search_docs", arguments={"query": "AI"}, ...)
    """
    warnings.warn(
        "tc() is deprecated. Use ToolCall() directly: ToolCall(search, query='AI')",
        DeprecationWarning,
        stacklevel=2,
    )

    if call_id is not None and (not isinstance(call_id, str) or call_id == ""):
        raise TypeError(f"tc(): call_id must be a non-empty str (got {call_id!r}).")

    coerced_arguments = _validate_json_serializable(arguments)

    name = _resolve_tool_name(tool, _name)

    return ToolCall(name=name, arguments=coerced_arguments, call_id=call_id)


def _resolve_tool_name(
    tool: str | Any | None,
    override: str | None,
    explicit_name: str | None = None,
) -> str:
    """Resolve tool name from object reference or string.

    Args:
        tool: Tool callable or name string.
        override: Explicit name override (takes precedence).
        explicit_name: Fallback name if tool is None.

    Returns:
        Resolved tool name string.

    Raises:
        TypeError: If tool name cannot be resolved.
    """
    if override is not None:
        return override

    if tool is None:
        if explicit_name is None:
            raise TypeError("ToolCall requires a tool name or callable")
        return explicit_name

    if isinstance(tool, str):
        return tool

    # Classes aren't callable endpoints - user likely meant an instance method
    if inspect.isclass(tool):
        raise TypeError(
            f"Got a class {tool.__name__}. Tools must be callable endpoints.\n"
            f"Did you mean {tool.__name__}.method or {tool.__name__}().__call__?"
        )

    if not callable(tool):
        raise TypeError(
            f"ToolCall() first argument must be str or callable, got {type(tool).__name__}"
        )

    if isinstance(tool, functools.partial):
        name: str = tool.func.__name__
        return name

    if isinstance(tool, classmethod):
        cm_name: str = tool.__func__.__name__
        return cm_name

    if hasattr(tool, "__name__"):
        fn_name: str = tool.__name__
        return fn_name

    if hasattr(tool, "__class__"):
        cls_name: str = tool.__class__.__name__
        return cls_name

    raise TypeError("ToolCall(): unable to infer tool name; pass name='...'")


def _coerce_argument(value: Any, key: str | None = None) -> tuple[Any, bool]:
    """Convert common Python types to JSON-serializable equivalents.

    Args:
        value: The value to coerce.
        key: Optional argument name for warning messages.

    Returns:
        Tuple of (coerced_value, was_coerced).

    Supported coercions:
    - Pydantic v2 BaseModel → model_dump(mode="json")
    - Pydantic v1 BaseModel → dict()
    - dataclasses → asdict()
    - datetime/date → isoformat()
    """
    original_type = type(value).__name__

    if hasattr(value, "model_dump"):
        coerced = value.model_dump(mode="json")
        _emit_coercion_warning(key, original_type, coerced)
        return coerced, True

    if hasattr(value, "__pydantic_model__") or (
        hasattr(value, "dict") and hasattr(value, "__fields__")
    ):
        coerced = value.dict()
        _emit_coercion_warning(key, original_type, coerced)
        return coerced, True

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        coerced = dataclasses.asdict(value)
        _emit_coercion_warning(key, original_type, coerced)
        return coerced, True

    if isinstance(value, (datetime.datetime, datetime.date)):
        coerced = value.isoformat()
        _emit_coercion_warning(key, original_type, coerced)
        return coerced, True

    return value, False


def _emit_coercion_warning(key: str | None, original_type: str, coerced: Any) -> None:
    """Emit a warning about automatic type coercion."""
    coerced_repr = repr(coerced)
    if len(coerced_repr) > 100:
        coerced_repr = coerced_repr[:97] + "..."

    key_part = f"'{key}'" if key else "value"
    # stacklevel=5: ToolCall.__init__ → _validate → _coerce → _emit → warn
    warn(
        f"ToolCall() coerced {key_part} from {original_type} to {coerced_repr}",
        TenroCoercionWarning,
        stacklevel=5,
    )


def _coerce_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Apply explicit coercions to all argument values."""
    return {k: _coerce_argument(v, k)[0] for k, v in arguments.items()}


def _validate_json_serializable(arguments: dict[str, Any]) -> dict[str, Any]:
    """Coerce and validate that arguments are JSON-serializable.

    Applies automatic coercions for Pydantic models, dataclasses, and datetime,
    then validates. Uses allow_nan=False to reject NaN/Infinity (not valid JSON).

    Returns:
        Coerced arguments dict.

    Raises:
        TypeError: If arguments cannot be serialized after coercion.
    """
    coerced = _coerce_arguments(arguments)
    try:
        json.dumps(coerced, allow_nan=False)
    except (TypeError, ValueError) as e:
        for key, value in coerced.items():
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError):
                type_name = type(arguments[key]).__name__
                raise TypeError(
                    f"ToolCall(): argument '{key}' is not JSON-serializable ({type_name}). "
                    "Supported: Pydantic models, dataclasses, datetime/date, JSON primitives."
                ) from e
        raise TypeError(f"ToolCall(): arguments are not JSON-serializable: {e}") from e
    return coerced


__all__ = ["ToolCall", "tc"]
