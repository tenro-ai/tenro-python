# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pattern: Simulating tool and LLM responses.

Shows how to control what your tools and LLMs return during tests.
"""

from examples.myapp import (
    SearchAgent,
    TopicConversationAgent,
    call_openai,
    search_documents,
)

from tenro import Construct, Provider
from tenro.simulate import llm, tool


def test_single_tool_result(construct: Construct) -> None:
    """Return the same value every time the tool is called."""
    # Control tool output - same result on every call
    tool.simulate(search_documents, result=["doc1", "doc2", "doc3"])

    # Tool returns simulated data instead of calling real vector DB
    SearchAgent().run("machine learning")

    tool.verify_many(search_documents, count=1)


def test_sequential_tool_results(construct: Construct) -> None:
    """Return different values on each successive call."""
    # First call returns empty, second returns results (simulate retry logic)
    tool.simulate(
        search_documents,
        results=[[], ["doc1", "doc2"]],  # Different result each call
    )

    # Each call gets the next result in sequence
    search_documents("first query")
    search_documents("retry query")

    # Verify both calls happened
    tool.verify_many(search_documents, count=2)


def test_single_llm_response(construct: Construct) -> None:
    """Control what the LLM returns."""
    # Every OpenAI call returns this response
    llm.simulate(Provider.OPENAI, response="The document discusses...")

    # Call the LLM function
    call_openai("What does the document say?")

    llm.verify(Provider.OPENAI, output_contains="discusses")


def test_sequential_llm_responses(construct: Construct) -> None:
    """Different responses for multi-turn conversations."""
    # Simulate a back-and-forth conversation
    llm.simulate(
        Provider.OPENAI,
        responses=[
            "Let me search for that.",
            "Based on the documents, the answer is...",
            "Is there anything else I can help with?",
        ],
    )

    # Run the conversation agent - each LLM call gets the next response
    TopicConversationAgent().run("machine learning")

    # Each LLM call gets the next response in the sequence
    llm.verify_many(Provider.OPENAI, at_least=1)
