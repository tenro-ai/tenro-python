# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Basic example: Research assistant agent.

Tests an agent that searches the web and summarizes findings.
"""

from __future__ import annotations

import tenro
from tenro import Provider, link_agent, link_llm, link_tool
from tenro.simulate import llm, tool

# APPLICATION CODE


@link_tool("web_search")
def web_search(query: str, num_results: int = 5) -> list[dict]:
    """Search the web for information."""
    return [{"title": "Result 1", "snippet": "...", "url": "https://..."}]


@link_llm(Provider.OPENAI)
def synthesize_findings(query: str, search_results: list[dict]) -> str:
    """Synthesize search results into a coherent summary."""
    import openai

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": f"Query: {query}\nResults: {search_results}\n\nSynthesize.",
            }
        ],
    )
    return response.choices[0].message.content


@link_agent("ResearchAssistantAgent")
class ResearchAssistantAgent:
    """Agent that researches a topic and provides a summary with sources."""

    def run(self, question: str) -> dict:
        """Run the research assistant agent."""
        results = web_search(question, num_results=5)
        summary = synthesize_findings(question, results)
        return {
            "summary": summary,
            "sources": [r["url"] for r in results],
        }


# TESTS


@tenro.simulate
def test_research_agent_finds_and_summarizes():
    """Test that agent searches and synthesizes results."""
    # Control what tools and LLMs return
    tool.simulate(
        web_search,
        result=[
            {
                "title": "AI Trends 2025",
                "snippet": "AI agents are...",
                "url": "https://example.com/1",
            },
            {"title": "Future of AI", "snippet": "Agentic AI...", "url": "https://example.com/2"},
        ],
    )
    llm.simulate(
        Provider.OPENAI,
        response="AI agents are becoming the dominant paradigm in 2025.",
    )

    # Run the agent
    ResearchAssistantAgent().run("What are the AI trends in 2025?")

    # Verify behavior
    tool.verify_many(web_search, count=1)
    llm.verify_many(Provider.OPENAI, at_least=1)


@tenro.simulate
def test_research_agent_handles_no_results():
    """Test agent behavior when search returns nothing."""
    # Simulate empty search results
    tool.simulate(web_search, result=[])
    llm.simulate(
        Provider.OPENAI,
        response="I couldn't find relevant information on this topic.",
    )

    # Run the agent
    ResearchAssistantAgent().run("Very obscure topic")

    # Verify graceful fallback
    tool.verify_many(web_search, count=1)
    llm.verify(output_contains="couldn't find")
