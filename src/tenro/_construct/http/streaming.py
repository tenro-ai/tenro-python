# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""SSE renderer for streaming LLM simulation.

Renders provider-shaped SSE bytes from fixture templates so the real
client SDK's stream parser reconstructs the response deterministically.
Plaintext streaming only; tool-call streaming is not yet supported.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import httpx

from tenro._construct.http.test_vectors.stream_renderer import render_stream
from tenro.errors.base import TenroError

SSE_CONTENT_TYPE = "text/event-stream; charset=utf-8"

_PROVIDER_CONFIG: dict[str, dict[str, str]] = {
    "openai": {
        "route": "chat_completions",
        "default_model": "gpt-4o",
        "id_prefix": "chatcmpl",
    },
    "anthropic": {
        "route": "messages",
        "default_model": "claude-3-5-sonnet-latest",
        "id_prefix": "msg",
    },
    "gemini": {
        "route": "generate_content",
        "default_model": "gemini-1.5-flash",
        "id_prefix": "gemini",
    },
}


def build_stream_bytes(
    provider: str,
    content: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    model: str | None = None,
    feature: str = "text",
    token_usage: dict[str, int] | None = None,
) -> bytes:
    r"""Render a provider's SSE byte stream for simulated content.

    Content, model, and token counts flow into the fixture as
    ``{{content}}``, ``{{model}}``, and ``{{usage_<field>}}`` token
    substitutions. Empty ``content`` omits the content-delta frame
    to match provider behavior.

    Args:
        provider: One of ``openai``, ``anthropic``, ``gemini``.
        content: Response text. Empty string omits the content delta.
        tool_calls: Reserved for tool-call streaming (not yet supported).
        model: Model name to echo; defaults per provider when ``None``.
        feature: Fixture feature name.
        token_usage: Token counts; surfaces as ``usage_<field>`` tokens.

    Returns:
        Concatenated SSE byte stream.

    Raises:
        TenroError: On tool-call streaming, unknown provider, or missing
            fixture.

    Example:
        ```python
        sse = build_stream_bytes("openai", "hello world")
        assert sse.endswith(b"data: [DONE]\n\n")
        ```

        With a custom model:
        ```python
        sse = build_stream_bytes("anthropic", "hi", model="claude-3-opus-latest")
        assert b"claude-3-opus-latest" in sse
        ```
    """
    if tool_calls:
        msg = (
            "Streaming simulation with tool calls is not yet supported. "
            "Use non-streaming mode for tool tests."
        )
        raise TenroError(msg)

    config = _PROVIDER_CONFIG.get(provider)
    if config is None:
        raise TenroError(f"Streaming not supported for provider {provider!r}")

    substitutions = _build_substitutions(content, model, config, token_usage)
    try:
        return render_stream(
            provider,
            config["route"],
            feature,
            substitutions=substitutions,
        )
    except FileNotFoundError as exc:
        raise TenroError(
            f"No streaming fixture for {provider}/{feature!r}; add a "
            f"{feature}_stream.json fixture or use a supported feature."
        ) from exc
    except ValueError as exc:
        raise TenroError(f"Invalid streaming fixture request: {exc}") from exc


def _build_substitutions(
    content: str,
    model: str | None,
    config: dict[str, str],
    token_usage: dict[str, int] | None,
) -> dict[str, Any]:
    """Assemble the per-call substitution map for fixture rendering."""
    usage = token_usage or {}
    default_output = max(1, len(content.split()) or 1)
    input_tokens = usage.get("input_tokens", 1)
    output_tokens = usage.get("output_tokens", default_output)
    now = int(time.time())
    return {
        "content": content,
        "model": model or config["default_model"],
        "id": f"{config['id_prefix']}-{uuid.uuid4().hex[:24]}",
        "created": now,
        "usage_input_tokens": input_tokens,
        "usage_output_tokens_start": usage.get("output_tokens_start", 1),
        "usage_output_tokens": output_tokens,
        "usage_total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
    }


def extract_token_usage(response_item: dict[str, Any], provider: str) -> dict[str, int] | None:
    """Normalize provider usage metadata into ``build_stream_bytes`` keys.

    OpenAI puts ``{prompt_tokens, completion_tokens, total_tokens}`` under
    ``usage``. Anthropic puts ``{input_tokens, output_tokens}`` under
    ``usage``. Gemini puts ``{promptTokenCount, candidatesTokenCount,
    totalTokenCount}`` under ``usageMetadata``. Returns ``None`` when
    usage data is absent.

    Args:
        response_item: Already-built non-streaming response JSON.
        provider: Provider name (``openai``, ``anthropic``, ``gemini``).

    Returns:
        Usage dict keyed by ``input_tokens`` / ``output_tokens`` /
        ``total_tokens``, or ``None`` when no usage is present.

    Example:
        ```python
        usage = extract_token_usage(
            {"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
            "openai",
        )
        assert usage == {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
        ```
    """
    if provider == "gemini":
        meta = response_item.get("usageMetadata")
        if not isinstance(meta, dict):
            return None
        return {
            "input_tokens": int(meta.get("promptTokenCount", 0)),
            "output_tokens": int(meta.get("candidatesTokenCount", 0)),
            "total_tokens": int(meta.get("totalTokenCount", 0)),
        }

    usage = response_item.get("usage")
    if not isinstance(usage, dict):
        return None
    if provider == "anthropic":
        return {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        }
    return {
        "input_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }


def build_stream_response(
    provider: str,
    content: str,
    response_item: dict[str, Any],
    tool_calls: list[dict[str, Any]],
) -> httpx.Response:
    """Return an SSE ``httpx.Response`` for the simulated content.

    Pulls ``model`` and ``usage`` out of the already-built non-streaming
    response so the streamed chunks carry the same metadata as the
    non-streaming path would have.
    """
    model = _extract_model(response_item, provider)
    token_usage = extract_token_usage(response_item, provider)
    stream_kwargs: dict[str, Any] = {}
    if model is not None:
        stream_kwargs["model"] = model
    if token_usage is not None:
        stream_kwargs["token_usage"] = token_usage
    sse_bytes = build_stream_bytes(provider, content, tool_calls=tool_calls, **stream_kwargs)
    return httpx.Response(
        200,
        headers={"content-type": SSE_CONTENT_TYPE},
        content=sse_bytes,
    )


def _extract_model(response_item: dict[str, Any], provider: str) -> str | None:
    """Pull the model name out of a simulated response JSON."""
    if provider == "gemini":
        model = response_item.get("modelVersion") or response_item.get("model")
    else:
        model = response_item.get("model")
    return model if isinstance(model, str) else None


__all__ = [
    "SSE_CONTENT_TYPE",
    "build_stream_bytes",
    "build_stream_response",
    "extract_token_usage",
]
