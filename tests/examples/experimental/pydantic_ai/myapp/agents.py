# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pydantic AI-based agents using OpenAI.

These agents demonstrate Pydantic AI patterns including:
- Agent for LLM orchestration
- @agent.tool decorator for tool registration
- RunContext for accessing dependencies
- Tool-based retrieval patterns
"""

from __future__ import annotations

from dataclasses import dataclass

from examples.myapp import fetch_documents, search_knowledge_base
from pydantic_ai import Agent, RunContext

from tenro import link_agent


@dataclass
class SupportDeps:
    """Dependencies for customer support agent."""

    question: str


@dataclass
class RAGDeps:
    """Dependencies for RAG pipeline."""

    topic: str


# Module-level agent with tools registered once
_support_agent: Agent[SupportDeps, str] = Agent(
    "openai:gpt-4o-mini",
    deps_type=SupportDeps,
    system_prompt=(
        "You are a helpful customer support agent. "
        "Use the search_kb tool to find relevant information before answering."
    ),
)


@_support_agent.tool
def search_kb(ctx: RunContext[SupportDeps]) -> str:
    """Search the knowledge base for information related to the question."""
    articles = search_knowledge_base(ctx.deps.question)
    return "\n".join(a["content"] for a in articles)


# Module-level RAG agent with tools registered once
_rag_agent: Agent[RAGDeps, str] = Agent(
    "openai:gpt-4o-mini",
    deps_type=RAGDeps,
    system_prompt=(
        "You are a research assistant. "
        "Use the fetch_docs tool to retrieve relevant documents, then answer the question."
    ),
)


@_rag_agent.tool
def fetch_docs(ctx: RunContext[RAGDeps]) -> str:
    """Fetch documents related to the topic."""
    documents = fetch_documents(ctx.deps.topic)
    return "\n".join(d["text"] for d in documents)


@link_agent("CustomerSupportAgent")
class CustomerSupportAgent:
    """Answer customer questions using Pydantic AI with @agent.tool."""

    def run(self, question: str) -> str:
        """Process a customer support question.

        Uses @agent.tool pattern for knowledge base retrieval.

        Args:
            question: Customer's question.

        Returns:
            Support response.
        """
        result = _support_agent.run_sync(question, deps=SupportDeps(question=question))
        return result.output


@link_agent("RAGPipeline")
class RAGPipeline:
    """Answer questions using Pydantic AI with @agent.tool for document retrieval."""

    def run(self, question: str, topic: str) -> str:
        """Process a question with RAG using tool-based retrieval.

        Uses @agent.tool pattern for document fetching.

        Args:
            question: User's question.
            topic: Topic to retrieve documents for.

        Returns:
            Synthesized answer.
        """
        result = _rag_agent.run_sync(question, deps=RAGDeps(topic=topic))
        return result.output


@link_agent("ConversationAgent")
class ConversationAgent:
    """Handle multi-turn conversations with Pydantic AI."""

    def run(self, user_messages: list[str]) -> list[str]:
        """Process multiple user messages maintaining context.

        Args:
            user_messages: List of user messages in order.

        Returns:
            List of assistant responses.
        """
        from pydantic_ai import Agent

        conversation_agent: Agent[None, str] = Agent(
            "openai:gpt-4o-mini",
            system_prompt="You are a helpful coding assistant.",
        )

        responses: list[str] = []

        for user_msg in user_messages:
            result = conversation_agent.run_sync(user_msg)
            responses.append(result.output)

        return responses
