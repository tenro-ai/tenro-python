# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Testing agents with LangChain's third-party tools.

Pattern: Use tenro.register() to enable simulation of third-party tools
without modifying application code.

This demonstrates how to test agents that use framework-provided tools
like DuckDuckGoSearchRun and WikipediaQueryRun without wrapping them
in your source code.
"""

from __future__ import annotations

import os

from examples.experimental.langchain.myapp.research_agent_anthropic import (
    WebResearchAgentAnthropic,
)
from examples.experimental.langchain.myapp.research_agent_gemini import (
    WebResearchAgentGemini,
)
from examples.experimental.langchain.myapp.research_agent_openai import (
    WebResearchAgentOpenAI,
)
from langchain_community import tools as lc_tools

import tenro
from tenro import Provider
from tenro.simulate import agent, llm, register, tool

# langchain-google-genai validates API key at init time (unlike OpenAI/Anthropic)
os.environ.setdefault("GOOGLE_API_KEY", "test-key-for-simulation")


@tenro.simulate
def test_research_agent_with_openai() -> None:
    """Test agent using DuckDuckGo and Wikipedia tools with OpenAI."""
    register(lc_tools.DuckDuckGoSearchRun.invoke)
    register(lc_tools.WikipediaQueryRun.invoke)

    tool.simulate(
        lc_tools.DuckDuckGoSearchRun.invoke,
        result="AI agents are software that perform tasks autonomously...",
    )
    tool.simulate(
        lc_tools.WikipediaQueryRun.invoke,
        result="Artificial intelligence agents perceive and act...",
    )
    llm.simulate(
        Provider.OPENAI,
        response="AI agents are autonomous systems that perceive and act.",
    )

    result = WebResearchAgentOpenAI().run("What are AI agents?")

    assert "AI" in result
    agent.verify(WebResearchAgentOpenAI)
    llm.verify(Provider.OPENAI)


@tenro.simulate
def test_research_agent_with_anthropic() -> None:
    """Test agent using DuckDuckGo and Wikipedia tools with Anthropic."""
    register(lc_tools.DuckDuckGoSearchRun.invoke)
    register(lc_tools.WikipediaQueryRun.invoke)

    tool.simulate(
        lc_tools.DuckDuckGoSearchRun.invoke,
        result="AI agents are software that perform tasks autonomously...",
    )
    tool.simulate(
        lc_tools.WikipediaQueryRun.invoke,
        result="Artificial intelligence agents perceive and act...",
    )
    llm.simulate(
        Provider.ANTHROPIC,
        response="AI agents are autonomous systems that perceive and act.",
    )

    result = WebResearchAgentAnthropic().run("What are AI agents?")

    assert "AI" in result
    agent.verify(WebResearchAgentAnthropic)
    llm.verify(Provider.ANTHROPIC)


@tenro.simulate
def test_research_agent_with_gemini() -> None:
    """Test agent using DuckDuckGo and Wikipedia tools with Gemini."""
    register(lc_tools.DuckDuckGoSearchRun.invoke)
    register(lc_tools.WikipediaQueryRun.invoke)

    tool.simulate(
        lc_tools.DuckDuckGoSearchRun.invoke,
        result="AI agents are software that perform tasks autonomously...",
    )
    tool.simulate(
        lc_tools.WikipediaQueryRun.invoke,
        result="Artificial intelligence agents perceive and act...",
    )
    llm.simulate(
        Provider.GEMINI,
        response="AI agents are autonomous systems that perceive and act.",
    )

    result = WebResearchAgentGemini().run("What are AI agents?")

    assert "AI" in result
    agent.verify(WebResearchAgentGemini)
    llm.verify(Provider.GEMINI)
