# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pattern: Verifying LLM response content.

Shows how to check what the LLM responded in multi-turn conversations.
"""

from __future__ import annotations

from examples.myapp import TopicConversationAgent

from tenro import Provider
from tenro.simulate import llm
from tenro.testing import tenro


@tenro
def test_check_first_response() -> None:
    """Verify content in the first LLM response (default)."""
    llm.simulate(
        Provider.OPENAI,
        responses=[
            "Machine learning is a subset of AI...",
            "For example, spam filters use ML to...",
            "You're welcome! Let me know if you have more questions.",
        ],
    )

    TopicConversationAgent().run("machine learning")

    # Default: checks the first response
    llm.verify(output_contains="Machine learning")


@tenro
def test_check_specific_response() -> None:
    """Verify content in a specific response by index."""
    llm.simulate(
        Provider.OPENAI,
        responses=[
            "First response",
            "Second response with ERROR",
            "Third response",
        ],
    )

    TopicConversationAgent().run("topic")

    # Verify the second simulated response contains ERROR
    # Use call_index=None to search any call for the content
    llm.verify(output_contains="ERROR", call_index=None)


@tenro
def test_check_last_response() -> None:
    """Verify content in the last response using negative indexing."""
    llm.simulate(
        Provider.OPENAI,
        responses=[
            "Starting...",
            "Processing...",
            "Task completed successfully!",
        ],
    )

    TopicConversationAgent().run("task")

    # Negative index: -1 is the last response
    llm.verify(output_contains="completed", call_index=-1)


@tenro
def test_check_any_response() -> None:
    """Verify content exists in any response (permissive mode)."""
    llm.simulate(
        Provider.OPENAI,
        responses=[
            "Looking into it...",
            "Found a critical security issue!",
            "Analysis complete.",
        ],
    )

    TopicConversationAgent().run("security audit")

    # call_index=None: matches if ANY response contains the text
    llm.verify(output_contains="security", call_index=None)
