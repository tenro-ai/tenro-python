# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""LlamaIndex-based agents using OpenAI.

These agents demonstrate LlamaIndex patterns including:
- Document and VectorStoreIndex for RAG
- Query engines for retrieval and synthesis
- SimpleChatEngine with ChatMemoryBuffer for conversations
- MockEmbedding for deterministic testing (no API calls)
"""

from __future__ import annotations

from examples.myapp import fetch_documents, search_knowledge_base

from tenro import link_agent


@link_agent("CustomerSupportAgent")
class CustomerSupportAgent:
    """Answer customer questions using LlamaIndex query engine."""

    def run(self, question: str) -> str:
        """Process a customer support question.

        Uses VectorStoreIndex with MockEmbedding for retrieval.

        Args:
            question: Customer's question.

        Returns:
            Support response.
        """
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.llms.openai import OpenAI

        # Configure LLM and embeddings (MockEmbedding for testing)
        Settings.llm = OpenAI(model="gpt-4o-mini")
        Settings.embed_model = MockEmbedding(embed_dim=1536)

        # Get context from knowledge base
        articles = search_knowledge_base(question)
        documents = [Document(text=a["content"]) for a in articles]

        # Build index and query
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        response = query_engine.query(question)

        return str(response)


@link_agent("RAGPipeline")
class RAGPipeline:
    """Answer questions using LlamaIndex RAG with query engine."""

    def run(self, question: str, topic: str) -> str:
        """Process a question with RAG.

        Uses VectorStoreIndex with MockEmbedding for retrieval.

        Args:
            question: The question to answer.
            topic: Topic to retrieve documents for.

        Returns:
            Synthesized answer.
        """
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.llms.openai import OpenAI

        # Configure LLM and embeddings (MockEmbedding for testing)
        Settings.llm = OpenAI(model="gpt-4o-mini")
        Settings.embed_model = MockEmbedding(embed_dim=1536)

        # Fetch documents
        docs = fetch_documents(topic)
        documents = [Document(text=d["text"]) for d in docs]

        # Build index and query
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        response = query_engine.query(question)

        return str(response)


@link_agent("ConversationAgent")
class ConversationAgent:
    """Handle multi-turn conversations with LlamaIndex SimpleChatEngine."""

    def run(self, user_messages: list[str]) -> list[str]:
        """Process multiple user messages using SimpleChatEngine with memory.

        Uses ChatMemoryBuffer for conversation history management.

        Args:
            user_messages: List of user messages in order.

        Returns:
            List of assistant responses.
        """
        from llama_index.core import Settings
        from llama_index.core.chat_engine import SimpleChatEngine
        from llama_index.core.memory import ChatMemoryBuffer
        from llama_index.llms.openai import OpenAI

        # Configure LLM
        llm = OpenAI(model="gpt-4o-mini")
        Settings.llm = llm

        # Create memory buffer for conversation history
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

        # Create chat engine with memory
        chat_engine = SimpleChatEngine.from_defaults(
            llm=llm,
            memory=memory,
            system_prompt="You are a helpful coding assistant.",
        )

        responses: list[str] = []

        for user_msg in user_messages:
            response = chat_engine.chat(user_msg)
            responses.append(str(response))

        return responses
