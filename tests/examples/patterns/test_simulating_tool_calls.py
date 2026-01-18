# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pattern: Simulating LLM tool calls with tc() helper.

Shows how to simulate LLM responses that include tool calls using the type-safe
tc() helper and ToolCall dataclass across different providers.

Key concepts:
- tc(func, **args): Create tool call from callable (type-safe, IDE autocomplete)
- ToolCall(name, arguments): Create tool call from string name (escape hatch)
- tool_calls=[...]: Attach tool calls to LLM response
- Per-response tool_calls in responses=[{...}] for multi-turn control
- Verification with llm.verify(), tool.verify(), llm.calls(), tool.calls()
"""

import unittest

from tenro import Construct, Provider, ToolCall, link_agent, link_llm, link_tool, tc
from tenro.simulate import llm, tool

# ============================================================================
# APPLICATION CODE - Tools
# ============================================================================


@link_tool("search")
def search(query: str, limit: int = 10) -> list[str]:
    """Search for results."""
    return []


@link_tool("get_weather")
def get_weather(city: str) -> dict:
    """Get weather for a city."""
    return {}


@link_tool("send_email")
def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email."""
    return True


# ============================================================================
# APPLICATION CODE - LLM Client Wrappers
# ============================================================================


@link_llm(Provider.OPENAI)
def call_openai(prompt: str) -> str:
    """Call OpenAI API."""
    import openai

    client = openai.OpenAI(api_key="test-key")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


@link_llm(Provider.ANTHROPIC)
def call_anthropic(prompt: str) -> str:
    """Call Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(api_key="test-key")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


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


# ============================================================================
# APPLICATION CODE - Class-based Agents
# ============================================================================


class SearchAgent:
    """Agent that searches and summarizes results using a single LLM provider."""

    def __init__(self, provider: str = "openai"):
        self.provider = provider

    @link_agent("SearchAgent")
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


class MultiTurnAgent:
    """Agent that has multi-turn conversations with the LLM."""

    @link_agent("MultiTurnAgent")
    def run(self, topic: str) -> str:
        """Execute multi-turn conversation."""
        r1 = call_openai(f"Start researching: {topic}")
        search("follow up query")
        r2 = call_openai("Continue with the findings")
        return f"{r1} | {r2}"


class DefensiveAgent:
    """Agent that handles unknown tool calls gracefully."""

    @link_agent("DefensiveAgent")
    def run(self, query: str) -> str:
        """Execute with defensive error handling."""
        response = call_openai(query)
        return response


# Instantiate agents for each provider
openai_search_agent = SearchAgent(provider="openai")
anthropic_search_agent = SearchAgent(provider="anthropic")
gemini_search_agent = SearchAgent(provider="gemini")

openai_multi_tool_agent = MultiToolAgent(provider="openai")
anthropic_multi_tool_agent = MultiToolAgent(provider="anthropic")
gemini_multi_tool_agent = MultiToolAgent(provider="gemini")

multi_turn_agent = MultiTurnAgent()
defensive_agent = DefensiveAgent()


# ============================================================================
# PYTEST - OpenAI Examples
# ============================================================================


def test_openai_single_tool_call(construct: Construct) -> None:
    """OpenAI: Single tool call using tc() helper."""
    llm.simulate(
        Provider.OPENAI,
        response="Summary of results",
        tool_calls=[tc(search, query="AI research")],
    )
    tool.simulate(search, result=["paper1", "paper2"])

    openai_search_agent.run("Find AI papers")

    llm.verify(Provider.OPENAI)
    tool.verify_many(search, count=1)

    assert len(llm.calls()) == 1
    assert len(tool.calls()) == 1
    assert tool.calls()[0].display_name == "search"


def test_openai_multiple_tool_calls(construct: Construct) -> None:
    """OpenAI: Multiple tool calls in parallel."""
    llm.simulate(
        Provider.OPENAI,
        response="Summary with weather",
        tool_calls=[
            tc(search, query="news"),
            tc(get_weather, city="NYC"),
        ],
    )
    tool.simulate(search, result=["headline"])
    tool.simulate(get_weather, result={"temp": 72})

    openai_multi_tool_agent.run("Get news and weather")

    tool.verify_many(count=2)
    tool.verify_many(search, count=1)
    tool.verify_many(get_weather, count=1)


def test_openai_multi_turn(construct: Construct) -> None:
    """OpenAI: Multi-turn with per-response tool_calls."""
    llm.simulate(
        Provider.OPENAI,
        responses=[
            {"text": "Starting search", "tool_calls": [tc(search, query="first")]},
            {"text": "Final summary"},
        ],
    )
    tool.simulate(search, result=["r1"])

    multi_turn_agent.run("research topic")

    llm.verify_many(Provider.OPENAI, count=2)
    tool.verify_many(search, count=1)
    assert len(llm.calls()) == 2


def test_openai_with_dataclass(construct: Construct) -> None:
    """OpenAI: ToolCall dataclass for string names."""
    llm.simulate(
        Provider.OPENAI,
        response="Found results",
        tool_calls=[ToolCall(name="search", arguments={"query": "AI"})],
    )
    tool.simulate(search, result=["result"])

    openai_search_agent.run("Search")

    tool.verify_many(search, count=1)
    llm.verify(Provider.OPENAI)


# ============================================================================
# PYTEST - Anthropic Examples
# ============================================================================


def test_anthropic_single_tool_call(construct: Construct) -> None:
    """Anthropic: Single tool call using tc() helper."""
    llm.simulate(
        Provider.ANTHROPIC,
        response="Summary of results",
        tool_calls=[tc(search, query="ML papers")],
    )
    tool.simulate(search, result=["paper1", "paper2"])

    anthropic_search_agent.run("Find ML papers")

    llm.verify(Provider.ANTHROPIC)
    tool.verify_many(search, count=1)

    assert len(llm.calls()) == 1
    assert llm.calls()[0].provider == "anthropic"


def test_anthropic_multiple_tool_calls(construct: Construct) -> None:
    """Anthropic: Multiple tool calls in a single response."""
    llm.simulate(
        Provider.ANTHROPIC,
        response="Results with weather",
        tool_calls=[
            tc(search, query="data"),
            tc(get_weather, city="Paris"),
        ],
    )
    tool.simulate(search, result=["doc1"])
    tool.simulate(get_weather, result={"temp": 20})

    anthropic_multi_tool_agent.run("Find data with weather")

    llm.verify(Provider.ANTHROPIC)
    tool.verify_many(count=2)


def test_anthropic_with_dataclass(construct: Construct) -> None:
    """Anthropic: ToolCall dataclass for string names."""
    llm.simulate(
        Provider.ANTHROPIC,
        response="Search complete",
        tool_calls=[ToolCall(name="search", arguments={"query": "docs"})],
    )
    tool.simulate(search, result=["doc1"])

    anthropic_search_agent.run("Find docs")

    tool.verify_many(search, count=1)
    llm.verify(Provider.ANTHROPIC)


# ============================================================================
# PYTEST - Gemini Examples
# ============================================================================


def test_gemini_single_tool_call(construct: Construct) -> None:
    """Gemini: Single tool call using tc() helper."""
    llm.simulate(
        Provider.GEMINI,
        response="Summary of results",
        tool_calls=[tc(search, query="AI trends")],
    )
    tool.simulate(search, result=["trend1", "trend2"])

    gemini_search_agent.run("Find AI trends")

    llm.verify(Provider.GEMINI)
    tool.verify_many(search, count=1)
    assert llm.calls()[0].provider == "gemini"


def test_gemini_multiple_tool_calls(construct: Construct) -> None:
    """Gemini: Multiple tool calls in a single response."""
    llm.simulate(
        Provider.GEMINI,
        response="Weather and search results",
        tool_calls=[
            tc(search, query="news"),
            tc(get_weather, city="Tokyo"),
        ],
    )
    tool.simulate(search, result=["news1"])
    tool.simulate(get_weather, result={"temp": 25})

    gemini_multi_tool_agent.run("Find news with weather")

    llm.verify(Provider.GEMINI)
    tool.verify_many(count=2)


def test_gemini_with_dataclass(construct: Construct) -> None:
    """Gemini: ToolCall dataclass for string names."""
    llm.simulate(
        Provider.GEMINI,
        response="Results found",
        tool_calls=[ToolCall(name="search", arguments={"query": "gemini"})],
    )
    tool.simulate(search, result=["result"])

    gemini_search_agent.run("Search gemini")

    tool.verify_many(search, count=1)
    llm.verify(Provider.GEMINI)


# ============================================================================
# UNITTEST - OpenAI Examples
# ============================================================================


class TestOpenAIToolCalls(unittest.TestCase):
    """Unittest examples for OpenAI tool calls."""

    def test_single_tool_call(self) -> None:
        """OpenAI unittest: Single tool call."""
        with Construct() as construct:
            llm.simulate(
                Provider.OPENAI,
                response="Results found",
                tool_calls=[tc(search, query="unittest query")],
            )
            tool.simulate(search, result=["result"])

            openai_search_agent.run("unittest search")

            tool.verify(search)
            construct.verify_llm(Provider.OPENAI)
            self.assertEqual(len(construct.tool_calls), 1)

    def test_multiple_tool_calls(self) -> None:
        """OpenAI unittest: Multiple tool calls."""
        with Construct() as construct:
            llm.simulate(
                Provider.OPENAI,
                response="Summary",
                tool_calls=[
                    tc(search, query="test"),
                    tc(get_weather, city="Boston"),
                ],
            )
            tool.simulate(search, result=["r"])
            tool.simulate(get_weather, result={"temp": 60})

            openai_multi_tool_agent.run("test task")

            construct.verify_tools(count=2)
            self.assertEqual(len(construct.llm_calls), 1)


# ============================================================================
# UNITTEST - Anthropic Examples
# ============================================================================


class TestAnthropicToolCalls(unittest.TestCase):
    """Unittest examples for Anthropic tool calls."""

    def test_single_tool_call(self) -> None:
        """Anthropic unittest: Single tool call."""
        with Construct() as construct:
            llm.simulate(
                Provider.ANTHROPIC,
                response="Found results",
                tool_calls=[tc(search, query="anthropic test")],
            )
            tool.simulate(search, result=["result"])

            anthropic_search_agent.run("search anthropic")

            tool.verify(search)
            construct.verify_llm(Provider.ANTHROPIC)

    def test_with_dataclass(self) -> None:
        """Anthropic unittest: ToolCall dataclass."""
        with Construct() as construct:
            llm.simulate(
                Provider.ANTHROPIC,
                response="Done",
                tool_calls=[ToolCall(name="search", arguments={"query": "data"})],
            )
            tool.simulate(search, result=["data1"])

            anthropic_search_agent.run("find data")

            tool.verify(search)
            self.assertEqual(construct.llm_calls[0].provider, "anthropic")


# ============================================================================
# UNITTEST - Gemini Examples
# ============================================================================


class TestGeminiToolCalls(unittest.TestCase):
    """Unittest examples for Gemini tool calls."""

    def test_single_tool_call(self) -> None:
        """Gemini unittest: Single tool call."""
        with Construct() as construct:
            llm.simulate(
                Provider.GEMINI,
                response="Summary ready",
                tool_calls=[tc(search, query="gemini test")],
            )
            tool.simulate(search, result=["result"])

            gemini_search_agent.run("search gemini")

            tool.verify(search)
            construct.verify_llm(Provider.GEMINI)

    def test_with_dataclass(self) -> None:
        """Gemini unittest: ToolCall dataclass."""
        with Construct() as construct:
            llm.simulate(
                Provider.GEMINI,
                response="Complete",
                tool_calls=[ToolCall(name="search", arguments={"query": "info"})],
            )
            tool.simulate(search, result=["info1"])

            gemini_search_agent.run("find info")

            tool.verify(search)
            self.assertEqual(construct.llm_calls[0].provider, "gemini")


# ============================================================================
# Simulating LLM Hallucinations and Invalid Tool Calls
# ============================================================================


def test_llm_requests_nonexistent_tool(construct: Construct) -> None:
    """Test agent handling of LLM requesting a tool that doesn't exist.

    This pattern is useful for testing defensive programming - ensuring your
    agent gracefully handles cases where the LLM hallucinates a tool name
    or requests a tool that isn't registered.
    """
    llm.simulate(
        Provider.OPENAI,
        response="I'll use the magic_tool to help",
        tool_calls=[ToolCall(name="nonexistent_magic_tool", arguments={"spell": "abracadabra"})],
    )

    defensive_agent.run("Do something magical")

    llm.verify(Provider.OPENAI)

    llm_call = llm.calls()[0]
    assert llm_call.response is not None


def test_llm_requests_tool_with_invalid_args(construct: Construct) -> None:
    """Test agent handling of LLM providing invalid arguments to a tool.

    Another defensive programming pattern - the LLM might call a real tool
    but with arguments that don't match the expected schema.
    """
    llm.simulate(
        Provider.OPENAI,
        response="Searching with limit",
        tool_calls=[ToolCall(name="search", arguments={"query": "test", "limit": "not_a_number"})],
    )
    tool.simulate(search, result=["result"])

    openai_search_agent.run("Search with bad args")

    llm.verify(Provider.OPENAI)
