# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""CrewAI-based agents using OpenAI.

These agents demonstrate CrewAI patterns including:
- Agent with role, goal, and backstory
- @tool decorator for tool integration
- Task with description and expected output
- Crew for orchestrating agents and tasks
- Memory enabled for context retention

Note: @link_tool is applied as the inner decorator (before @tool) so that
CrewAI captures the Tenro-wrapped function. When CrewAI executes the tool,
it calls through Tenro's wrapper, enabling simulation interception.
"""

from __future__ import annotations

from crewai import Agent, Crew, Task
from crewai.tools import tool
from examples.myapp import fetch_documents, search_knowledge_base

from tenro import link_agent, link_tool


@tool("Search Knowledge Base")
@link_tool("search_kb")
def search_kb(query: str) -> str:
    """Search the company knowledge base for relevant articles.

    Args:
        query: Search query for the knowledge base.

    Returns:
        Relevant knowledge base content.
    """
    articles = search_knowledge_base(query)
    return "\n".join(a["content"] for a in articles)


@tool("Fetch Documents")
@link_tool("fetch_docs")
def fetch_docs(topic_query: str) -> str:
    """Fetch documents from the document store.

    Args:
        topic_query: Topic to search for.

    Returns:
        Document contents.
    """
    documents = fetch_documents(topic_query)
    return "\n".join(d["text"] for d in documents)


@link_agent("CustomerSupportAgent")
class CustomerSupportAgent:
    """Answer customer questions using CrewAI with tool integration."""

    def run(self, question: str) -> str:
        """Process a customer support question."""
        support_agent = Agent(
            role="Customer Support Representative",
            goal="Help customers by searching the knowledge base and providing accurate answers",
            backstory="Experienced support agent with access to the knowledge base.",
            tools=[search_kb],
            llm="gpt-4o-mini",
            memory=True,
        )

        support_task = Task(
            description=f"Answer this customer question: {question}",
            expected_output="A helpful, accurate response based on knowledge base information",
            agent=support_agent,
        )

        crew = Crew(agents=[support_agent], tasks=[support_task])
        result = crew.kickoff()
        return str(result)


@link_agent("RAGPipeline")
class RAGPipeline:
    """Answer questions using CrewAI with tool-based document retrieval."""

    def run(self, question: str, topic: str) -> str:
        """Process a question with RAG using CrewAI tools."""
        researcher = Agent(
            role="Document Analyst",
            goal="Retrieve and analyze documents to answer questions accurately",
            backstory="Expert researcher who uses document retrieval tools.",
            tools=[fetch_docs],
            llm="gpt-4o-mini",
            memory=True,
        )

        analysis_task = Task(
            description=(
                f"First use the fetch_docs tool to get documents about '{topic}', "
                f"then answer this question: {question}"
            ),
            expected_output="A comprehensive answer based on the retrieved documents",
            agent=researcher,
        )

        crew = Crew(agents=[researcher], tasks=[analysis_task])
        result = crew.kickoff()
        return str(result)


@link_agent("ConversationAgent")
class ConversationAgent:
    """Handle multi-turn conversations with CrewAI memory."""

    def run(self, user_messages: list[str]) -> list[str]:
        """Process multiple user messages with memory-enabled agent."""
        assistant = Agent(
            role="Coding Assistant",
            goal="Help users with programming questions, remembering previous context",
            backstory="You are an expert programmer who maintains conversation context.",
            llm="gpt-4o-mini",
            memory=True,
        )

        responses: list[str] = []
        conversation_context: list[str] = []

        for user_msg in user_messages:
            conversation_context.append(f"User: {user_msg}")

            history = "\n".join(conversation_context)
            task = Task(
                description=(
                    f"Given this conversation history:\n{history}\n\n"
                    "Respond to the latest user message."
                ),
                expected_output="A helpful response continuing the conversation",
                agent=assistant,
            )

            crew = Crew(
                agents=[assistant],
                tasks=[task],
                memory=True,
            )
            result = crew.kickoff()
            response = str(result)

            conversation_context.append(f"Assistant: {response}")
            responses.append(response)

        return responses
