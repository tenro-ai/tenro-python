# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Canonical serialization for trace policy transform helpers."""

from __future__ import annotations

import base64
import json
from typing import Any

_SUPPORTED_TYPES_MSG = (
    "supported types: None, bool, int, float, str, bytes, dict, list, tuple, set, frozenset"
)


def to_canonical_bytes(value: Any) -> bytes:
    """Return a deterministic, type-preserving UTF-8 encoding of ``value``.

    Every value is wrapped in a ``[type_tag, payload]`` pair before JSON
    serialisation so distinct Python values never collide — including
    strings that happen to look like other encoded types.

    Raises:
        TypeError: If ``value`` or any nested element is an unsupported
            type, or if a dict contains a non-string key.
    """
    return _compact_json(_tag(value))


def to_display_bytes(value: Any) -> bytes:
    """Return a best-effort human-readable UTF-8 encoding of ``value``.

    Strings encode as their own UTF-8 bytes; ``bytes`` / ``bytearray`` are
    base64-wrapped with a ``b64:`` marker; every other value is serialised
    as compact JSON. Lossy: non-string dict keys
    are stringified (so ``1`` and ``"1"`` collapse) and arbitrary objects
    fall back to ``repr()`` (which may leak memory addresses or other
    implementation details). Intended for preview helpers (``truncate``,
    ``length``) only — never use for hashing or equality.
    """
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray)):
        return b"b64:" + base64.b64encode(bytes(value))
    return _compact_json(_display_coerce(value))


def _compact_json(value: Any) -> bytes:
    """Serialise ``value`` as deterministic, compact UTF-8 JSON bytes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _tag(value: Any) -> list[Any]:
    """Return ``[type_tag, payload]`` for ``value`` — type-preserving form.

    Every accepted runtime type maps to a distinct tag (``bytes`` vs
    ``bytearray``, ``set`` vs ``frozenset``, ``list`` vs ``tuple``) so
    aliased pairs never produce the same canonical bytes.
    """
    # bool must be checked before int since bool is a subclass of int.
    if value is None:
        return ["none", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, (int, float, str)):
        return [type(value).__name__, value]
    if isinstance(value, bytes):
        return ["bytes_b64", base64.b64encode(value).decode("ascii")]
    if isinstance(value, bytearray):
        return ["bytearray_b64", base64.b64encode(bytes(value)).decode("ascii")]
    return _tag_container(value)


def _tag_container(value: Any) -> list[Any]:
    """Tag container types (dict/set/frozenset/list/tuple) with distinct tags."""
    if isinstance(value, dict):
        return ["dict", _tag_dict(value)]
    if isinstance(value, frozenset):
        return ["frozenset", sorted((_tag(x) for x in value), key=json.dumps)]
    if isinstance(value, set):
        return ["set", sorted((_tag(x) for x in value), key=json.dumps)]
    if isinstance(value, tuple):
        return ["tuple", [_tag(x) for x in value]]
    if isinstance(value, list):
        return ["list", [_tag(x) for x in value]]
    raise TypeError(
        f"cannot canonicalize value of type {type(value).__name__!r}; {_SUPPORTED_TYPES_MSG}"
    )


def _tag_dict(value: dict[Any, Any]) -> list[list[Any]]:
    """Tag a dict as a sorted list of ``[tagged_key, tagged_value]`` pairs.

    Keys must be strings so canonical bytes stay deterministic — integer
    or object keys would sort unstably across Python versions.
    """
    pairs: list[list[Any]] = []
    for key, val in value.items():
        if not isinstance(key, str):
            raise TypeError(
                f"dict keys must be str for canonical encoding, got {type(key).__name__!r}"
            )
        pairs.append([_tag(key), _tag(val)])
    pairs.sort(key=lambda pair: pair[0][1])
    return pairs


def _display_coerce(value: Any) -> Any:
    """Make ``value`` JSON-serialisable for display encoding.

    Never raises. Dicts with non-string keys become an ordered list of
    ``[stringified_key, value]`` pairs. Unsupported objects emit a bounded
    ``"<ClassName>"`` marker.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return "b64:" + base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, dict):
        return [[str(key), _display_coerce(val)] for key, val in value.items()]
    if isinstance(value, (set, frozenset)):
        return sorted((_display_coerce(x) for x in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_display_coerce(x) for x in value]
    return f"<{type(value).__name__}>"


__all__ = ["to_canonical_bytes", "to_display_bytes"]
