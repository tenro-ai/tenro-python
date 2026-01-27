# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""RAG Pipeline: Testing document retrieval with custom OpenAI agents."""

from __future__ import annotations

from examples.myapp import RAGPipeline, fetch_documents, generate_response

from tenro import Provider
from tenro.simulate import llm, tool
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
    llm.simulate(
        Provider.OPENAI,
        target=generate_response,
        response="Machine learning is a field where algorithms learn patterns from data.",
    )

    RAGPipeline().run("What is machine learning?", "AI")

    tool.verify_many(fetch_documents, count=1)
    llm.verify(Provider.OPENAI)
    llm.verify(output_contains="Machine learning")
