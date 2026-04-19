# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Render streaming SSE fixtures into provider byte output."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_TOKEN_PATTERN = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


def render_stream(
    provider: str,
    route: str,
    feature: str = "text",
    *,
    substitutions: dict[str, Any] | None = None,
) -> bytes:
    r"""Render a streaming fixture into SSE byte output.

    Substitution is single-pass — replacements cannot trigger further
    replacement, so caller content containing ``{{token}}`` survives
    untouched. Events may declare a ``skip_if_empty`` list; if any
    listed key resolves to an empty string, the event is omitted to
    match provider behavior for empty deltas.

    Args:
        provider: Provider name (``openai``, ``anthropic``, ``gemini``).
        route: API route (``chat_completions``, ``messages``, ``generate_content``).
        feature: Feature name (``text``, ``tool_calls``).
        substitutions: Mapping of token name to replacement value.

    Returns:
        Concatenated SSE byte stream ready to return as an HTTP body.

    Raises:
        FileNotFoundError: If the streaming fixture is missing.
        ValueError: If any path segment contains separators or traversal.

    Example:
        ```python
        sse = render_stream(
            "anthropic",
            "messages",
            "text",
            substitutions={"content": "hi claude", "model": "claude-3-5-sonnet-latest"},
        )
        assert b"event: message_stop" in sse
        ```
    """
    fixture = _load_stream_fixture(provider, route, feature)
    events: list[dict[str, Any]] = fixture.get("events", [])
    out: list[bytes] = []
    for event in events:
        if substitutions is not None and _should_skip(event, substitutions):
            continue
        data = _substitute(event.get("data"), substitutions or {})
        out.append(_format_frame(data, event.get("event")))
    return b"".join(out)


def _should_skip(event: dict[str, Any], substitutions: dict[str, Any]) -> bool:
    """Return True when any ``skip_if_empty`` key is present and empty.

    A missing key leaves its ``{{token}}`` marker intact (per ``_substitute``)
    and does not trigger a skip — skipping only fires when the caller
    explicitly supplies an empty value for that key.
    """
    skip_keys = event.get("skip_if_empty") or []
    return any(key in substitutions and substitutions[key] == "" for key in skip_keys)


def _load_stream_fixture(provider: str, route: str, feature: str) -> dict[str, Any]:
    """Load a streaming SSE event fixture from disk.

    Rejects path separators and traversal components in any segment so
    callers cannot resolve fixtures outside the package root.
    """
    for name, segment in (("provider", provider), ("route", route), ("feature", feature)):
        if "/" in segment or "\\" in segment or segment in ("", ".", ".."):
            raise ValueError(f"Invalid {name} segment: {segment!r}")

    package_root = Path(__file__).parent.resolve()
    fixture_path = (package_root / provider / route / f"{feature}_stream.json").resolve()

    if not fixture_path.is_relative_to(package_root):
        raise ValueError("Invalid fixture path: escapes package root")

    if not fixture_path.exists():
        raise FileNotFoundError(f"Streaming fixture not found: {fixture_path}")

    with open(fixture_path) as f:
        loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Fixture {fixture_path} must be a JSON object")
        return loaded


_WHOLE_TOKEN_PATTERN = re.compile(r"^\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}$")


def _substitute(value: Any, substitutions: dict[str, Any]) -> Any:
    """Recursively replace ``{{key}}`` tokens in string values.

    Single-pass replacement — values splice in literally so caller
    content containing ``{{token}}`` round-trips untouched. A leaf
    that is exactly ``"{{key}}"`` yields the substitution value
    with its original type, preserving JSON numeric fields like
    ``usage.output_tokens``. Missing keys leave their marker intact.
    """
    if isinstance(value, str):
        whole = _WHOLE_TOKEN_PATTERN.match(value)
        if whole is not None and whole.group(1) in substitutions:
            return substitutions[whole.group(1)]
        return _TOKEN_PATTERN.sub(
            lambda m: str(substitutions[m.group(1)]) if m.group(1) in substitutions else m.group(0),
            value,
        )
    if isinstance(value, dict):
        return {k: _substitute(v, substitutions) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, substitutions) for v in value]
    return value


def _format_frame(data: Any, event: str | None) -> bytes:
    """Format a single SSE frame as bytes.

    String ``data`` is emitted verbatim (e.g. ``"[DONE]"``); dicts
    are JSON-encoded compactly. Frames are terminated by the double
    newline required by the SSE spec.
    """
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    if event is not None:
        return f"event: {event}\ndata: {payload}\n\n".encode()
    return f"data: {payload}\n\n".encode()


__all__ = ["render_stream"]
