# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""RAG Pipeline: Testing document retrieval with AutoGen."""

from __future__ import annotations

from examples.experimental.autogen.myapp.agents import RAGPipeline, fetch_docs

import tenro
from tenro import Provider, ToolCall
from tenro.simulate import agent, llm, tool


@tenro.simulate
def test_rag_pipeline_synthesizes_answer() -> None:
    """Test RAG pipeline fetches documents and generates answer.

    Flow: LLM decides to call fetch_docs → tool returns docs → LLM synthesizes.
    """
    # Simulate the tool that LLM will call
    tool.simulate(
        fetch_docs,
        result="Machine learning uses algorithms to learn.\nDeep learning is a subset of ML.",
    )

    # Simulate LLM: first requests tool, then synthesizes answer
    llm.simulate(
        Provider.OPENAI,
        responses=[
            [ToolCall("fetch_docs", topic="AI")],
            "ML uses algorithms to learn from data. TERMINATE",
        ],
    )

    result = RAGPipeline().run("What is machine learning?", "AI")

    assert result == "ML uses algorithms to learn from data. TERMINATE"
    agent.verify(RAGPipeline)
    llm.verify_many(Provider.OPENAI, count=2)
    tool.verify_many(fetch_docs, count=1)
