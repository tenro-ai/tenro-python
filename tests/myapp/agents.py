# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Sample agents and tools for testing."""

from __future__ import annotations

import openai

from tenro import link_agent, link_llm, link_tool


@link_llm
def chat_completion(prompt: str) -> str:
    """Make a chat completion call."""
    client = openai.OpenAI(api_key="test-key")
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


@link_tool
def search(query: str) -> list[str]:
    """Search tool."""
    raise NotImplementedError("Should be simulated")


@link_tool
def fetch(url: str) -> str:
    """Fetch tool."""
    raise NotImplementedError("Should be simulated")


@link_agent
def researcher(task: str) -> str:
    """Research agent."""
    raise NotImplementedError("Should be simulated")


@link_agent
def writer(prompt: str) -> str:
    """Writer agent."""
    raise NotImplementedError("Should be simulated")
