# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""RAG Pipeline: Testing document retrieval with LangGraph."""

from __future__ import annotations

from examples.experimental.langgraph.myapp.agents import RAGPipeline
from examples.myapp import fetch_documents

import tenro
from tenro import Provider
from tenro.simulate import agent, llm, tool


@tenro.simulate
def test_rag_pipeline_synthesizes_answer() -> None:
    """Test RAG pipeline fetches documents and generates answer."""
    tool.simulate(
        fetch_documents,
        result=[
            {"id": "doc1", "text": "Machine learning uses algorithms to learn."},
            {"id": "doc2", "text": "Deep learning is a subset of ML."},
        ],
    )
    llm.simulate(
        Provider.OPENAI,
        response="Machine learning is a field where algorithms learn patterns from data.",
    )

    result = RAGPipeline().run("What is machine learning?", "AI")

    assert result == "Machine learning is a field where algorithms learn patterns from data."
    agent.verify(RAGPipeline)
    llm.verify(Provider.OPENAI)
    tool.verify_many(fetch_documents, count=1)
