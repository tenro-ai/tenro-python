# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Advanced example: RAG document agent.

Tests a complete RAG (Retrieval-Augmented Generation) pipeline:
1. Vector search for relevant documents
2. Fetch full document content
3. Rerank by relevance
4. Generate answer with context
"""

from tenro import Construct, Provider, link_agent, link_llm, link_tool
from tenro.simulate import llm, tool

# APPLICATION CODE


@link_tool("vector_search")
def vector_search(query: str, top_k: int = 3) -> list[str]:
    """Search vector database for relevant document IDs."""
    return ["doc_001", "doc_002", "doc_003"]


@link_tool("fetch_documents")
def fetch_documents(doc_ids: list[str]) -> list[dict]:
    """Fetch full documents by their IDs."""
    return [{"id": doc_id, "content": f"Content of {doc_id}"} for doc_id in doc_ids]


@link_llm(Provider.OPENAI)
def generate_answer(context: str, question: str) -> str:
    """Generate an answer using the context."""
    import openai

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


@link_llm(Provider.OPENAI)
def rerank_documents(question: str, documents: list[dict]) -> list[dict]:
    """Use LLM to rerank documents by relevance."""
    import openai

    openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": f"Rank these by relevance to '{question}': {documents}",
            }
        ],
    )
    return documents


@link_agent("RAGDocumentAgent")
class RAGDocumentAgent:
    """Complete RAG pipeline."""

    def run(self, question: str) -> dict:
        """Run the RAG document agent."""
        doc_ids = vector_search(question, top_k=5)
        documents = fetch_documents(doc_ids)
        ranked_docs = rerank_documents(question, documents)
        context = "\n".join(doc["content"] for doc in ranked_docs[:3])
        answer = generate_answer(context, question)

        return {
            "question": question,
            "answer": answer,
            "sources": [doc["id"] for doc in ranked_docs[:3]],
        }


# TESTS


def test_rag_pipeline_answers_question(construct: Construct):
    """Test the complete RAG workflow."""
    # Control what tools and LLMs return
    construct.simulate_tool(
        vector_search,
        result=["doc_ai_safety", "doc_alignment", "doc_ethics"],
    )
    construct.simulate_tool(
        fetch_documents,
        result=[
            {"id": "doc_ai_safety", "content": "AI safety is about..."},
            {"id": "doc_alignment", "content": "Alignment ensures AI..."},
            {"id": "doc_ethics", "content": "Ethics in AI covers..."},
        ],
    )
    construct.simulate_llm(
        Provider.OPENAI,
        responses=[
            "Reranked documents.",
            "AI safety ensures systems behave as intended.",
        ],
    )

    # Run the agent
    RAGDocumentAgent().run("What is AI safety?")

    # Verify full pipeline ran
    tool.verify_many(vector_search, count=1)
    tool.verify_many(fetch_documents, count=1)
    llm.verify_many(Provider.OPENAI, at_least=2)


def test_rag_handles_no_results(construct: Construct):
    """Test RAG behavior when search returns nothing."""
    # Simulate empty search results
    construct.simulate_tool(vector_search, result=[])
    construct.simulate_tool(fetch_documents, result=[])
    construct.simulate_llm(
        Provider.OPENAI,
        responses=[
            "No documents to rerank.",
            "I don't have information about that topic.",
        ],
    )

    # Run the agent
    RAGDocumentAgent().run("What is quantum gravity?")

    # Verify graceful fallback
    tool.verify_many(vector_search, count=1)
    llm.verify(output_contains="don't have information")


def test_rag_handles_database_failure(construct: Construct):
    """Test RAG when document fetch fails."""
    # Inject database error
    construct.simulate_tool(vector_search, result=["doc1", "doc2"])
    construct.simulate_tool(
        fetch_documents,
        result=ConnectionError("Database unavailable"),
    )

    # Run the agent (expect error)
    import contextlib

    with contextlib.suppress(ConnectionError):
        RAGDocumentAgent().run("test query")

    # Verify partial execution
    tool.verify_many(vector_search, count=1)
    tool.verify_many(fetch_documents, count=1)
