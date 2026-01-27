# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""RAG Pipeline: Testing document retrieval with Pydantic AI."""

from __future__ import annotations

from examples.experimental.pydantic_ai.myapp.agents import RAGPipeline
from examples.myapp import fetch_documents

from tenro import Provider, ToolCall
from tenro.simulate import agent, llm, tool
from tenro.testing import tenro


@tenro
def test_rag_pipeline_synthesizes_answer() -> None:
    """Test RAG pipeline fetches documents and generates answer."""
    tool.simulate(
        fetch_documents,
        result=[
            {"id": "doc1", "text": "Machine learning uses algorithms to learn."},
            {"id": "doc2", "text": "Deep learning is a subset of ML."},
        ],
    )
    # Simulate LLM requesting the tool, then returning final answer
    llm.simulate(
        Provider.OPENAI,
        responses=[
            [ToolCall("fetch_docs")],
            "Machine learning is a field where algorithms learn patterns from data.",
        ],
    )

    result = RAGPipeline().run("What is machine learning?", "AI")

    assert result == "Machine learning is a field where algorithms learn patterns from data."
    agent.verify(RAGPipeline)
    llm.verify_many(Provider.OPENAI, count=2)
    tool.verify_many(fetch_documents, count=1)
