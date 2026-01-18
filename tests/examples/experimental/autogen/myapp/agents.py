# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""AutoGen-based agents using OpenAI.

These agents demonstrate AutoGen AgentChat patterns with AssistantAgent.
Tools are passed directly as async functions - no FunctionTool wrapper needed.
"""

from __future__ import annotations

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from tenro import link_agent, link_tool


# Define tools as module-level async functions
# AutoGen uses the function.__name__ as the tool name
@link_tool
async def search_kb(query: str) -> str:
    """Search the knowledge base for customer support information.

    Args:
        query: Search query for the knowledge base.

    Returns:
        Relevant knowledge base content.
    """
    # Production: query vector DB or search API
    return "Full refunds within 30 days."


@link_tool
async def fetch_docs(topic: str) -> str:
    """Fetch documents related to a topic.

    Args:
        topic: Topic to search for documents.

    Returns:
        Document contents.
    """
    # Production: query document store
    return "Machine learning uses algorithms to learn.\nDeep learning is a subset of ML."


@link_agent("CustomerSupportAgent")
class CustomerSupportAgent:
    """Answer customer questions using AutoGen with tool calling."""

    def run(self, question: str) -> str:
        """Process a customer support question.

        Args:
            question: Customer's question.

        Returns:
            Support response.
        """
        return asyncio.run(self._run_async(question))

    async def _run_async(self, question: str) -> str:
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key="test-key",
        )

        agent = AssistantAgent(
            name="support_agent",
            model_client=model_client,
            tools=[search_kb],
            system_message=(
                "You are a customer support agent. "
                "Use the search_kb tool to find relevant information. "
                "Say TERMINATE when done."
            ),
        )

        termination = TextMentionTermination("TERMINATE")
        team = RoundRobinGroupChat([agent], termination_condition=termination)

        result = await team.run(task=question)
        return str(result.messages[-1].content)


@link_agent("RAGPipeline")
class RAGPipeline:
    """Answer questions using document retrieval and synthesis."""

    def run(self, question: str, topic: str) -> str:
        """Process a question with RAG.

        Args:
            question: User's question.
            topic: Topic to retrieve documents for (unused - LLM decides).

        Returns:
            Synthesized answer.
        """
        return asyncio.run(self._run_async(question))

    async def _run_async(self, question: str) -> str:
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key="test-key",
        )

        agent = AssistantAgent(
            name="rag_agent",
            model_client=model_client,
            tools=[fetch_docs],
            system_message=(
                "You are a research assistant. "
                "Use the fetch_docs tool to retrieve documents, then answer the question. "
                "Say TERMINATE when done."
            ),
        )

        termination = TextMentionTermination("TERMINATE")
        team = RoundRobinGroupChat([agent], termination_condition=termination)

        result = await team.run(task=question)
        return str(result.messages[-1].content)


@link_agent("ConversationAgent")
class ConversationAgent:
    """Handle multi-turn conversations with AutoGen."""

    def run(self, user_messages: list[str]) -> list[str]:
        """Process multiple user messages maintaining context.

        Args:
            user_messages: List of user messages in order.

        Returns:
            List of assistant responses.
        """
        return asyncio.run(self._run_async(user_messages))

    async def _run_async(self, user_messages: list[str]) -> list[str]:
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key="test-key",
        )

        agent = AssistantAgent(
            name="coding_agent",
            model_client=model_client,
            system_message=(
                "You are a helpful coding assistant. Say TERMINATE when done with each response."
            ),
        )

        termination = TextMentionTermination("TERMINATE")
        team = RoundRobinGroupChat([agent], termination_condition=termination)

        responses: list[str] = []
        for user_msg in user_messages:
            result = await team.run(task=user_msg)
            responses.append(str(result.messages[-1].content))

        return responses
