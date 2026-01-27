# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""LLM function fixtures requiring client libraries.

These are separated from agents.py to avoid import errors in clean-room tests.
"""

from __future__ import annotations

import openai

from tenro import link_llm


@link_llm
def chat_completion(prompt: str) -> str:
    """Make a chat completion call."""
    client = openai.OpenAI(api_key="test-key")
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""
