# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Request parsing helpers for the HTTP interceptor."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from tenro.providers import Provider


def is_streaming_request(request: httpx.Request, provider: str) -> bool:
    """Detect whether the client requested a streamed response.

    OpenAI/Anthropic set ``stream: true`` in the JSON body. Gemini
    signals streaming via the ``:streamGenerateContent`` URL path.
    """
    if provider == Provider.GEMINI:
        return ":streamGenerateContent" in request.url.path
    try:
        body = json.loads(request.content)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(body, dict) and bool(body.get("stream"))


def parse_request_body(
    request: httpx.Request, provider: str
) -> tuple[list[dict[str, Any]], str | None]:
    """Extract messages and model from the request body.

    OpenAI/Anthropic use ``messages`` in the JSON body; Gemini uses
    ``contents`` and encodes the model name in the URL path. Returns
    empty/None values when the body is absent or malformed.
    """
    try:
        body = json.loads(request.content)
    except (json.JSONDecodeError, TypeError):
        return [], None

    if not isinstance(body, dict):
        return [], None

    messages = body.get("messages") or body.get("contents", [])
    model = body.get("model")

    if model is None and provider == Provider.GEMINI:
        match = re.search(r"/models/([^/:]+)", request.url.path)
        if match:
            model = match.group(1)

    return messages, model


__all__ = ["is_streaming_request", "parse_request_body"]
