# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""RAG Pipeline: Testing document retrieval with CrewAI.

NOTE: CrewAI uses ReAct text parsing, not OpenAI's native `tool_calls` structure.
Unlike LangChain/OpenAI SDK, CrewAI expects "Action:" and "Action Input:" in the
LLM's text output. Simulated responses must match this format.
"""

import json

from examples.experimental.crewai.myapp.agents import RAGPipeline, fetch_docs

from tenro import Provider
from tenro.simulate import agent, llm, tool


def react_action(tool_name: str, tool_input: dict) -> str:
    """Format a ReAct action that CrewAI's parser understands."""
    return (
        "Thought: I should fetch relevant documents.\n"
        f"Action: {tool_name}\n"
        f"Action Input: {json.dumps(tool_input)}\n"
    )


def react_final(answer: str) -> str:
    """Format a ReAct final answer that CrewAI's parser understands."""
    return f"Thought: I now know the final answer.\nFinal Answer: {answer}\n"


def test_rag_pipeline_synthesizes_answer(construct) -> None:
    """Test RAG pipeline fetches documents and generates answer."""
    tool.simulate(
        fetch_docs.func,
        result="Machine learning uses algorithms to learn.\nDeep learning is a subset of ML.",
    )

    llm.simulate(
        Provider.OPENAI,
        responses=[
            react_action("Fetch Documents", {"topic_query": "AI"}),
            react_final("Machine learning is a field where algorithms learn patterns."),
        ],
    )

    result = RAGPipeline().run("What is machine learning?", "AI")

    assert result == "Machine learning is a field where algorithms learn patterns."
    agent.verify(RAGPipeline)
    llm.verify_many(Provider.OPENAI, count=2)
    tool.verify_many(fetch_docs.func, count=1)
