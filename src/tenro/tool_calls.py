# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Tool call types and helpers for LLM simulation."""

from __future__ import annotations

import dataclasses
import datetime
import functools
import inspect
import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from tenro.errors.warnings import TenroCoercionWarning, warn

JSONPrimitive = str | int | float | bool | None
JSONValue = JSONPrimitive | dict[str, "JSONValue"] | list["JSONValue"]
JSONObject = dict[str, JSONValue]


@dataclass(frozen=True)
class ToolCall:
    """Provider-agnostic tool call representation.

    Args:
        name: Tool name (e.g., "get_weather"). Must be non-empty string.
        arguments: Tool arguments as dict. Must be JSON-serializable.
        call_id: Correlation ID for matching calls to results. Auto-generated if None.

    Returns:
        ToolCall instance.

    Examples:
        >>> ToolCall(name="search", arguments={"query": "AI"})
        ToolCall(name='search', arguments={'query': 'AI'}, call_id='call_...')

        >>> ToolCall(name="get_time")  # Zero arguments
        ToolCall(name='get_time', arguments={}, call_id='call_...')
    """

    name: str
    arguments: JSONObject = field(default_factory=dict)
    call_id: str | None = None

    def __post_init__(self) -> None:
        """Validate fields and auto-generate call_id if not provided."""
        if not isinstance(self.name, str):
            raise TypeError(f"name must be str, got {type(self.name).__name__}")
        if self.name == "":
            raise TypeError("name must be non-empty str")

        if not isinstance(self.arguments, dict):
            raise TypeError(f"arguments must be dict, got {type(self.arguments).__name__}")

        try:
            json.dumps(self.arguments, allow_nan=False)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"arguments must be JSON-serializable: {e}. Use tc() for automatic coercions."
            ) from e

        if self.call_id is not None:
            if not isinstance(self.call_id, str):
                raise TypeError(f"call_id must be str, got {type(self.call_id).__name__}")
            if self.call_id == "":
                raise TypeError("call_id must be non-empty str (got '')")
        else:
            object.__setattr__(self, "call_id", f"call_{uuid4().hex[:12]}")


def tc(
    tool: str | Any,
    *,
    _name: str | None = None,
    call_id: str | None = None,
    **arguments: Any,
) -> ToolCall:
    """Create a ToolCall from a tool name or callable.

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
    if call_id is not None and (not isinstance(call_id, str) or call_id == ""):
        raise TypeError(f"tc(): call_id must be a non-empty str (got {call_id!r}).")

    coerced_arguments = _validate_json_serializable(arguments)

    name = _resolve_tool_name(tool, _name)

    return ToolCall(name=name, arguments=coerced_arguments, call_id=call_id)


def _resolve_tool_name(tool: str | Any, override: str | None) -> str:
    """Resolve tool name from object reference or string.

    This extracts the BARE function name (e.g., "search") for LLM tool call
    representation. This is intentionally different from target_resolution.py
    which extracts full qualified paths for simulation patching.

    Priority:
    1. _name= override if provided
    2. String tool passed directly
    3. Callable: extract __name__ (rejects class types, handles partial/classmethod)
    """
    if override is not None:
        return override

    if isinstance(tool, str):
        return tool

    # Classes aren't callable endpoints - user likely meant an instance method
    if inspect.isclass(tool):
        raise TypeError(
            f"Got a class {tool.__name__}. Tools must be callable endpoints.\n"
            f"Did you mean {tool.__name__}.method or {tool.__name__}().__call__?"
        )

    if not callable(tool):
        raise TypeError(f"tc() first argument must be str or callable, got {type(tool).__name__}")

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

    raise TypeError("tc(): unable to infer tool name; pass _name='...'")


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
    # stacklevel=5: tc -> _validate_json_serializable -> _coerce_arguments -> here -> warn
    warn(
        f"tc() coerced {key_part} from {original_type} to {coerced_repr}",
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
                    f"tc(): argument '{key}' is not JSON-serializable ({type_name}). "
                    "Supported: Pydantic models, dataclasses, datetime/date, JSON primitives."
                ) from e
        raise TypeError(f"tc(): arguments are not JSON-serializable: {e}") from e
    return coerced


__all__ = ["ToolCall", "tc"]
