# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Transform helpers for ``capture=CUSTOM`` trace policies.

Each helper returns a ``Transform`` callable. Compose them directly or
wrap them per category:

    from tenro.trace_policy import configure_trace_policy, CaptureMode
    from tenro.trace_policy.transforms import keyed_blake2b, truncate

    hash_prompts = keyed_blake2b(key=b"project-secret")
    trim_outputs = truncate(head=512, tail=64)

    def policy(value, ctx):
        if ctx.category.value == "inputs":
            return hash_prompts(value, ctx)
        if ctx.category.value == "outputs":
            return trim_outputs(value, ctx)
        return value

    configure_trace_policy(capture=CaptureMode.CUSTOM, transform=policy)
"""

from __future__ import annotations

import hashlib
from typing import Any

from tenro.trace_policy._canonical import to_canonical_bytes, to_display_bytes
from tenro.trace_policy.modes import OMIT, Transform, TransformContext

_DIGEST_BYTES = 32  # BLAKE2b-256
_BLAKE2B_KEY_MIN = 1
_BLAKE2B_KEY_MAX = 64
_PSEUDONYM_HEX_CHARS = 16


def keyed_blake2b(key: bytes) -> Transform:
    """Return a transform that emits a keyed BLAKE2b-256 digest.

    Dict key order does not affect the digest — ``{"a": 1, "b": 2}`` and
    ``{"b": 2, "a": 1}`` produce the same output. Rotate the key to
    invalidate existing digests.

    Args:
        key: Secret key, 1..64 bytes (BLAKE2b keyed-mode limit).

    Returns:
        Transform emitting a 64-character hex digest.

    Raises:
        ValueError: If ``key`` length is outside the BLAKE2b keyed range.
        TypeError: If ``key`` is not bytes-like.
    """
    normalized = _normalize_key(key)

    def _hash(value: Any, _ctx: TransformContext) -> str:
        return _keyed_digest(normalized, value)

    _hash.__qualname__ = "transforms.keyed_blake2b"
    return _hash


def pseudonymize(key: bytes, *, prefix: str = "ps_") -> Transform:
    """Return a transform that emits a short, prefixed pseudonym.

    Useful for identifiers (user IDs, session IDs) where you need to join
    traces without exposing the raw value. Truncates the digest to 16 hex chars.

    Args:
        key: Secret key for keyed hashing.
        prefix: Human-readable prefix on the output. Defaults to ``"ps_"``.

    Returns:
        Transform emitting ``f"{prefix}{16_hex_chars}"``.
    """
    normalized = _normalize_key(key)

    def _pseudo(value: Any, _ctx: TransformContext) -> str:
        return f"{prefix}{_keyed_digest(normalized, value)[:_PSEUDONYM_HEX_CHARS]}"

    _pseudo.__qualname__ = "transforms.pseudonymize"
    return _pseudo


def truncate(head: int = 1024, tail: int = 128) -> Transform:
    """Return a transform that keeps only the head and tail of a value.

    Operates on the human-readable display encoding (strings pass through
    as UTF-8; structured values become compact sort-key JSON) so output
    is stable and legible. Inserts ``"...[+NB]..."`` between the kept
    parts, where ``N`` is the number of bytes dropped.

    Args:
        head: Bytes kept from the start. Must be non-negative.
        tail: Bytes kept from the end. Must be non-negative.

    Returns:
        Transform emitting a string.

    Raises:
        ValueError: If ``head`` or ``tail`` is negative.
    """
    if head < 0 or tail < 0:
        raise ValueError(f"head/tail must be non-negative, got head={head}, tail={tail}")

    def _truncate(value: Any, _ctx: TransformContext) -> str:
        raw = to_display_bytes(value)
        total = len(raw)
        if total <= head + tail:
            return raw.decode("utf-8", errors="replace")
        dropped = total - head - tail
        head_part = raw[:head].decode("utf-8", errors="replace")
        tail_part = raw[total - tail :].decode("utf-8", errors="replace") if tail else ""
        return f"{head_part}...[+{dropped}B]...{tail_part}"

    _truncate.__qualname__ = "transforms.truncate"
    return _truncate


def length(unit: str = "utf8_bytes") -> Transform:
    """Return a transform that emits only the length of a value.

    Args:
        unit: ``"utf8_bytes"`` (default) or ``"chars"``.

    Returns:
        Transform emitting an int.

    Raises:
        ValueError: If ``unit`` is unknown.
    """
    if unit not in ("utf8_bytes", "chars"):
        raise ValueError(f"unit must be 'utf8_bytes' or 'chars', got {unit!r}")

    def _length(value: Any, _ctx: TransformContext) -> int:
        if isinstance(value, str):
            return len(value) if unit == "chars" else len(value.encode("utf-8"))
        return len(to_display_bytes(value))

    _length.__qualname__ = "transforms.length"
    return _length


def drop_always(_value: Any, _ctx: TransformContext) -> Any:
    """Transform that always drops the value. Equivalent to group disable."""
    return OMIT


def _keyed_digest(key: bytes, value: Any) -> str:
    """Return the hex BLAKE2b-256 keyed digest of ``value``'s canonical bytes."""
    return hashlib.blake2b(
        to_canonical_bytes(value), digest_size=_DIGEST_BYTES, key=key
    ).hexdigest()


def _normalize_key(key: bytes) -> bytes:
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError(f"key must be bytes or bytearray, got {type(key).__name__}")
    if not _BLAKE2B_KEY_MIN <= len(key) <= _BLAKE2B_KEY_MAX:
        raise ValueError(
            f"key length must be {_BLAKE2B_KEY_MIN}..{_BLAKE2B_KEY_MAX} bytes "
            f"(BLAKE2b keyed-mode limit), got {len(key)}"
        )
    return bytes(key)


__all__ = ["drop_always", "keyed_blake2b", "length", "pseudonymize", "truncate"]
