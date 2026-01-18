# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""LlamaIndex-based agents using Google Gemini.

These agents demonstrate LlamaIndex patterns with Google's Gemini models.
"""

from __future__ import annotations

from examples.myapp import fetch_documents, search_knowledge_base

from tenro import link_agent


@link_agent("CustomerSupportAgentGemini")
class CustomerSupportAgentGemini:
    """Answer customer questions using LlamaIndex with Gemini."""

    def run(self, question: str) -> str:
        """Process a customer support question."""
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.llms.gemini import Gemini

        Settings.llm = Gemini(model="models/gemini-1.5-flash")
        Settings.embed_model = MockEmbedding(embed_dim=1536)

        articles = search_knowledge_base(question)
        documents = [Document(text=a["content"]) for a in articles]

        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        response = query_engine.query(question)

        return str(response)


@link_agent("RAGPipelineGemini")
class RAGPipelineGemini:
    """Answer questions using LlamaIndex RAG with Gemini."""

    def run(self, question: str, topic: str) -> str:
        """Process a question with RAG."""
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.llms.gemini import Gemini

        Settings.llm = Gemini(model="models/gemini-1.5-flash")
        Settings.embed_model = MockEmbedding(embed_dim=1536)

        docs = fetch_documents(topic)
        documents = [Document(text=d["text"]) for d in docs]

        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        response = query_engine.query(question)

        return str(response)


@link_agent("ConversationAgentGemini")
class ConversationAgentGemini:
    """Handle multi-turn conversations with LlamaIndex using Gemini."""

    def run(self, user_messages: list[str]) -> list[str]:
        """Process multiple user messages maintaining context."""
        from llama_index.core import Settings
        from llama_index.core.chat_engine import SimpleChatEngine
        from llama_index.core.memory import ChatMemoryBuffer
        from llama_index.llms.gemini import Gemini

        llm = Gemini(model="models/gemini-1.5-flash")
        Settings.llm = llm

        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

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
