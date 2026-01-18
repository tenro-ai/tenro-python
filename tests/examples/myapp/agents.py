# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Shared agents for example applications.

Contains LLM wrappers and agents used by the example tests.
This simulates a real application that would be tested with Tenro.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import openai
from examples.myapp.tools import (
    call_api,
    delete_all_records,
    fetch_documents,
    fetch_from_api,
    get_cached_data,
    get_weather,
    process_data,
    save_result,
    search,
    search_database,
    search_documents,
    search_knowledge_base,
    validate_input,
)

from tenro import Provider, link_agent, link_llm

# =============================================================================
# LLM WRAPPERS (from custom + patterns)
# =============================================================================


@link_llm(Provider.OPENAI)
def generate_response(system_prompt: str, user_message: str) -> str:
    """Generate a response using OpenAI.

    Args:
        system_prompt: System context for the model.
        user_message: User's question or input.

    Returns:
        Model's response text.
    """
    client = openai.OpenAI(api_key="test-key")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""


@link_llm(Provider.OPENAI)
def call_openai(prompt: str) -> str:
    """Call OpenAI API."""
    response = openai.OpenAI(api_key="test-key").chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


@link_llm(Provider.OPENAI)
def chat(message: str) -> str:
    """Simple chat interface to OpenAI."""
    response = openai.OpenAI(api_key="test-key").chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message.content or ""


@link_llm(Provider.OPENAI)
def call_llm(prompt: str) -> str:
    """Generic LLM call."""
    response = openai.OpenAI(api_key="test-key").chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


@link_llm(Provider.ANTHROPIC)
def call_anthropic(prompt: str) -> str:
    """Call Anthropic API."""
    import anthropic

    response = anthropic.Anthropic(api_key="test-key").messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text if response.content else ""


@link_llm(Provider.GEMINI)
def call_gemini(prompt: str) -> str:
    """Call Gemini API."""
    from google import genai

    client = genai.Client(api_key="test-key")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text or ""


@link_llm(Provider.OPENAI)
def chat_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict[str, Any]:
    """Call OpenAI chat API with optional tools.

    Returns dict with 'content' and optional 'tool_calls'.
    """
    client = openai.OpenAI(api_key="test-key")
    kwargs: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools

    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message

    result: dict[str, Any] = {"content": msg.content or ""}
    if msg.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            }
            for tc in msg.tool_calls
        ]
    return result


# =============================================================================
# AGENTS FROM custom/myapp/agents.py
# =============================================================================


@link_agent
class CustomerSupportAgent:
    """Answer customer questions using knowledge base and LLM."""

    def run(self, question: str) -> str:
        """Process a customer support question.

        Args:
            question: Customer's question.

        Returns:
            Support response.
        """
        articles = search_knowledge_base(question)
        context = "\n".join(a["content"] for a in articles)
        return generate_response(
            f"Use this context to help customers:\n{context}",
            question,
        )


@link_agent
class RAGPipeline:
    """Answer questions using document retrieval and synthesis."""

    def run(self, question: str, topic: str) -> str:
        """Process a question with RAG.

        Args:
            question: User's question.
            topic: Topic to retrieve documents for.

        Returns:
            Synthesized answer.
        """
        documents = fetch_documents(topic)
        docs_text = "\n".join(d["text"] for d in documents)
        return generate_response(
            f"Answer based on these documents:\n{docs_text}",
            question,
        )


# =============================================================================
# AGENTS FROM patterns/myapp/agents.py
# =============================================================================


@link_agent
class WeatherAgent:
    """Agent that answers weather questions using LLM-driven tool calls."""

    TOOLS: ClassVar[list[dict[str, Any]]] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    def run(self, question: str) -> str:
        """Answer a weather question. LLM decides whether to call tools."""
        messages = [
            {"role": "system", "content": "You help with weather. Use get_weather tool."},
            {"role": "user", "content": question},
        ]

        response = chat_with_tools(messages, tools=self.TOOLS)

        # If LLM requested tools, execute them and get final answer
        if "tool_calls" in response:
            messages.append(
                {"role": "assistant", "content": "", "tool_calls": response["tool_calls"]}
            )

            for tc in response["tool_calls"]:
                if tc["name"] == "get_weather":
                    result = get_weather(tc["arguments"]["city"])
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps(result),
                        }
                    )

            final = chat_with_tools(messages)
            return final["content"]

        return response["content"]


@link_agent
class DataAgent:
    """Agent that searches database using LLM-driven tool calls."""

    TOOLS: ClassVar[list[dict[str, Any]]] = [
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search database for records",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    def run(self, question: str) -> str:
        """Answer a question using database search."""
        messages = [
            {"role": "system", "content": "Search the database to answer questions."},
            {"role": "user", "content": question},
        ]

        response = chat_with_tools(messages, tools=self.TOOLS)

        if "tool_calls" in response:
            messages.append(
                {"role": "assistant", "content": "", "tool_calls": response["tool_calls"]}
            )

            for tc in response["tool_calls"]:
                if tc["name"] == "search_database":
                    result = search_database(tc["arguments"]["query"])
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps(result),
                        }
                    )

            final = chat_with_tools(messages)
            return final["content"]

        return response["content"]


@link_agent
class ResilientAgent:
    """Agent with retry logic for API failures."""

    def run(self, task: str) -> dict:
        """Execute task with retry on failure."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = call_api("/data")
                return {"success": True, "data": result}
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise
        return {"success": False}


@link_agent("SearchAgent")
class SearchAgent:
    """Agent that searches and summarizes documents."""

    def run(self, query: str) -> str:
        """Search and summarize documents."""
        docs = search_documents(query)
        return f"Found {len(docs)} documents"


@link_agent
class ConversationAgent:
    """Handle multi-turn conversations with history.

    This agent maintains conversation history and processes a list of user
    messages, returning a list of responses (one per message).
    """

    def run(self, user_messages: list[str]) -> list[str]:
        """Process multiple user messages maintaining context.

        Args:
            user_messages: List of user messages in order.

        Returns:
            List of assistant responses.
        """
        client = openai.OpenAI(api_key="test-key")
        messages: list[dict] = [
            {"role": "system", "content": "You are a helpful coding assistant."}
        ]
        responses: list[str] = []

        for user_msg in user_messages:
            messages.append({"role": "user", "content": user_msg})
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            assistant_msg = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": assistant_msg})
            responses.append(assistant_msg)

        return responses


@link_agent("TopicConversationAgent")
class TopicConversationAgent:
    """Agent that has a multi-turn conversation about a topic.

    This agent makes 3 hardcoded LLM calls to simulate a conversation.
    Used for testing verifying content patterns.
    """

    def run(self, topic: str) -> list[str]:
        """Have a multi-turn conversation about a topic."""
        responses = []
        responses.append(chat(f"Tell me about {topic}"))
        responses.append(chat("Tell me more"))
        responses.append(chat("Thanks!"))
        return responses


@link_agent("SmartCacheAgent", entry_points="get_data")
class SmartCacheAgent:
    """Agent that checks cache before calling API."""

    def get_data(self, key: str) -> dict:
        """Get data from cache or API."""
        cached = get_cached_data(key)
        if cached is not None:
            return cached
        return fetch_from_api(key)


@link_agent("SafeCleanupAgent", entry_points="cleanup")
class SafeCleanupAgent:
    """Agent that only deletes records when explicitly confirmed."""

    def cleanup(self, confirmed: bool = False) -> str:
        """Cleanup records if confirmed."""
        if confirmed:
            delete_all_records()
            return "Deleted all records"
        return "Cleanup cancelled - confirmation required"


@link_agent
class ValidationAgent:
    """Agent that validates input data."""

    def run(self, data: dict) -> bool:
        """Validate input."""
        return validate_input(data)


@link_agent
class ProcessingAgent:
    """Agent that processes data."""

    def run(self, data: dict) -> dict:
        """Process data."""
        return process_data(data)


@link_agent
class PersistenceAgent:
    """Agent that saves results."""

    def run(self, result: dict) -> str:
        """Save result."""
        return save_result(result)


@link_agent
class PipelineAgent:
    """Agent that orchestrates a data pipeline."""

    def run(self, data: dict) -> str:
        """Run the full pipeline: validate -> process -> save."""
        if validate_input(data):
            processed = process_data(data)
            return save_result(processed)
        return "validation-failed"


# =============================================================================
# MULTI-PROVIDER AGENTS (for tool call simulation tests)
# =============================================================================


class ProviderSearchAgent:
    """Agent that searches and summarizes results using a configurable LLM provider."""

    def __init__(self, provider: str = "openai"):
        self.provider = provider

    @link_agent("ProviderSearchAgent")
    def run(self, query: str) -> str:
        """Execute search and summarize."""
        results = search(query)
        prompt = f"Summarize these results: {results}"

        if self.provider == "openai":
            return call_openai(prompt)
        elif self.provider == "anthropic":
            return call_anthropic(prompt)
        else:
            return call_gemini(prompt)


class MultiToolAgent:
    """Agent that uses multiple tools (search + weather) with configurable LLM."""

    def __init__(self, provider: str = "openai"):
        self.provider = provider

    @link_agent("MultiToolAgent")
    def run(self, task: str) -> str:
        """Execute task using multiple tools."""
        results = search(task)
        weather = get_weather("NYC")
        prompt = f"Task: {task}\nSearch results: {results}\nWeather: {weather}"

        if self.provider == "openai":
            return call_openai(prompt)
        elif self.provider == "anthropic":
            return call_anthropic(prompt)
        else:
            return call_gemini(prompt)


@link_agent("MultiTurnAgent")
class MultiTurnAgent:
    """Agent that has multi-turn conversations with the LLM."""

    def run(self, topic: str) -> str:
        """Execute multi-turn conversation."""
        r1 = call_openai(f"Start researching: {topic}")
        search("follow up query")
        r2 = call_openai("Continue with the findings")
        return f"{r1} | {r2}"


@link_agent("DefensiveAgent")
class DefensiveAgent:
    """Agent that handles unknown tool calls gracefully."""

    def run(self, query: str) -> str:
        """Execute with defensive error handling."""
        response = call_openai(query)
        return response


# =============================================================================
# HELPER FUNCTIONS (non-decorated)
# =============================================================================


def get_data_with_cache(key: str, use_cache: bool = True) -> dict:
    """Get data using cache if available."""
    from examples.myapp.tools import check_cache

    if use_cache:
        cached = check_cache(key)
        if cached is not None:
            return cached
    return fetch_from_api(key)


def multi_step_workflow(topic: str) -> list[str]:
    """A workflow that makes multiple LLM calls."""
    responses = []
    responses.append(call_llm(f"Research {topic}"))
    responses.append(call_llm("Summarize findings"))
    responses.append(call_llm("Generate recommendations"))
    return responses


# =============================================================================
# AGENT INSTANCES (for convenience)
# =============================================================================

openai_search_agent = ProviderSearchAgent(provider="openai")
anthropic_search_agent = ProviderSearchAgent(provider="anthropic")
gemini_search_agent = ProviderSearchAgent(provider="gemini")

openai_multi_tool_agent = MultiToolAgent(provider="openai")
anthropic_multi_tool_agent = MultiToolAgent(provider="anthropic")
gemini_multi_tool_agent = MultiToolAgent(provider="gemini")

multi_turn_agent = MultiTurnAgent()
defensive_agent = DefensiveAgent()
