# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pattern: Simulating LLM tool calls with ToolCall and LLMResponse.

Shows how to simulate LLM responses that include tool calls using the type-safe
ToolCall constructor across different providers.

Key concepts:
- ToolCall(func, **args): Create tool call from callable (type-safe, IDE autocomplete)
- ToolCall("name", **args): Create tool call from string name
- responses=[ToolCall(...)]: Single tool call response
- responses=[["text", ToolCall(...)]]: Text + tool call in one response (nested list)
- responses=[LLMResponse([...])] : Explicit ordered blocks (interleaving)
- Verification with llm.verify(), tool.verify(), llm.calls(), tool.calls()

Interleaving note:
  Text blocks in LLMResponse represent the model's reasoning BEFORE/WHILE making
  tool call requests - NOT commentary on tool results. Tool results arrive in a
  separate response after tools execute. Anthropic and Gemini preserve block order;
  OpenAI Chat flattens to content + tool_calls (order lost but still works).
"""

from __future__ import annotations

import unittest

import tenro
from tenro import Construct, LLMResponse, Provider, ToolCall, link_agent, link_llm, link_tool
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
    from examples.myapp.clients import get_openai_client

    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


@link_llm(Provider.ANTHROPIC)
def call_anthropic(prompt: str) -> str:
    """Call Anthropic API."""
    from anthropic.types import TextBlock
    from examples.myapp.clients import get_anthropic_client

    client = get_anthropic_client()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text_blocks = [b for b in response.content if isinstance(b, TextBlock)]
    return text_blocks[0].text if text_blocks else ""


@link_llm(Provider.GEMINI)
def call_gemini(prompt: str) -> str:
    """Call Gemini API."""
    from examples.myapp.clients import get_gemini_client

    client = get_gemini_client()
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


@tenro.simulate
def test_openai_single_tool_call() -> None:
    """OpenAI: Single tool call using ToolCall() helper."""
    llm.simulate(
        Provider.OPENAI,
        responses=[["Summary of results", ToolCall(search, query="AI research")]],
    )
    tool.simulate(search, result=["paper1", "paper2"])

    openai_search_agent.run("Find AI papers")

    llm.verify(Provider.OPENAI)
    tool.verify_many(search, count=1)

    assert len(llm.calls()) == 1
    assert len(tool.calls()) == 1
    assert tool.calls()[0].display_name == "search"


@tenro.simulate
def test_openai_multiple_tool_calls() -> None:
    """OpenAI: Multiple tool calls in parallel."""
    llm.simulate(
        Provider.OPENAI,
        responses=[
            [
                "Summary with weather",
                ToolCall(search, query="news"),
                ToolCall(get_weather, city="NYC"),
            ]
        ],
    )
    tool.simulate(search, result=["headline"])
    tool.simulate(get_weather, result={"temp": 72})

    openai_multi_tool_agent.run("Get news and weather")

    tool.verify_many(count=2)
    tool.verify_many(search, count=1)
    tool.verify_many(get_weather, count=1)


@tenro.simulate
def test_openai_multi_turn() -> None:
    """OpenAI: Multi-turn with per-response tool calls."""
    llm.simulate(
        Provider.OPENAI,
        responses=[
            ["Starting search", ToolCall(search, query="first")],
            "Final summary",
        ],
    )
    tool.simulate(search, result=["r1"])

    multi_turn_agent.run("research topic")

    llm.verify_many(Provider.OPENAI, count=2)
    tool.verify_many(search, count=1)
    assert len(llm.calls()) == 2


@tenro.simulate
def test_openai_with_string_name() -> None:
    """OpenAI: ToolCall with string name."""
    llm.simulate(
        Provider.OPENAI,
        responses=[["Found results", ToolCall(name="search", arguments={"query": "AI"})]],
    )
    tool.simulate(search, result=["result"])

    openai_search_agent.run("Search")

    tool.verify_many(search, count=1)
    llm.verify(Provider.OPENAI)


# ============================================================================
# PYTEST - Anthropic Examples
# ============================================================================


@tenro.simulate
def test_anthropic_single_tool_call() -> None:
    """Anthropic: Single tool call using ToolCall() helper."""
    llm.simulate(
        Provider.ANTHROPIC,
        responses=[["Summary of results", ToolCall(search, query="ML papers")]],
    )
    tool.simulate(search, result=["paper1", "paper2"])

    anthropic_search_agent.run("Find ML papers")

    llm.verify(Provider.ANTHROPIC)
    tool.verify_many(search, count=1)

    assert len(llm.calls()) == 1
    assert llm.calls()[0].provider == "anthropic"


@tenro.simulate
def test_anthropic_multiple_tool_calls() -> None:
    """Anthropic: Multiple tool calls in a single response."""
    llm.simulate(
        Provider.ANTHROPIC,
        responses=[
            [
                "Results with weather",
                ToolCall(search, query="data"),
                ToolCall(get_weather, city="Paris"),
            ]
        ],
    )
    tool.simulate(search, result=["doc1"])
    tool.simulate(get_weather, result={"temp": 20})

    anthropic_multi_tool_agent.run("Find data with weather")

    llm.verify(Provider.ANTHROPIC)
    tool.verify_many(count=2)


@tenro.simulate
def test_anthropic_with_string_name() -> None:
    """Anthropic: ToolCall with string name."""
    llm.simulate(
        Provider.ANTHROPIC,
        responses=[["Search complete", ToolCall(name="search", arguments={"query": "docs"})]],
    )
    tool.simulate(search, result=["doc1"])

    anthropic_search_agent.run("Find docs")

    tool.verify_many(search, count=1)
    llm.verify(Provider.ANTHROPIC)


# ============================================================================
# PYTEST - Gemini Examples
# ============================================================================


@tenro.simulate
def test_gemini_single_tool_call() -> None:
    """Gemini: Single tool call using ToolCall() helper."""
    llm.simulate(
        Provider.GEMINI,
        responses=[["Summary of results", ToolCall(search, query="AI trends")]],
    )
    tool.simulate(search, result=["trend1", "trend2"])

    gemini_search_agent.run("Find AI trends")

    llm.verify(Provider.GEMINI)
    tool.verify_many(search, count=1)
    assert llm.calls()[0].provider == "gemini"


@tenro.simulate
def test_gemini_multiple_tool_calls() -> None:
    """Gemini: Multiple tool calls in a single response."""
    llm.simulate(
        Provider.GEMINI,
        responses=[
            [
                "Weather and search results",
                ToolCall(search, query="news"),
                ToolCall(get_weather, city="Tokyo"),
            ]
        ],
    )
    tool.simulate(search, result=["news1"])
    tool.simulate(get_weather, result={"temp": 25})

    gemini_multi_tool_agent.run("Find news with weather")

    llm.verify(Provider.GEMINI)
    tool.verify_many(count=2)


@tenro.simulate
def test_gemini_with_string_name() -> None:
    """Gemini: ToolCall with string name."""
    llm.simulate(
        Provider.GEMINI,
        responses=[["Results found", ToolCall(name="search", arguments={"query": "gemini"})]],
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
                responses=[["Results found", ToolCall(search, query="unittest query")]],
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
                responses=[
                    [
                        "Summary",
                        ToolCall(search, query="test"),
                        ToolCall(get_weather, city="Boston"),
                    ]
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
                responses=[["Found results", ToolCall(search, query="anthropic test")]],
            )
            tool.simulate(search, result=["result"])

            anthropic_search_agent.run("search anthropic")

            tool.verify(search)
            construct.verify_llm(Provider.ANTHROPIC)

    def test_with_string_name(self) -> None:
        """Anthropic unittest: ToolCall with string name."""
        with Construct() as construct:
            llm.simulate(
                Provider.ANTHROPIC,
                responses=[["Done", ToolCall(name="search", arguments={"query": "data"})]],
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
                responses=[["Summary ready", ToolCall(search, query="gemini test")]],
            )
            tool.simulate(search, result=["result"])

            gemini_search_agent.run("search gemini")

            tool.verify(search)
            construct.verify_llm(Provider.GEMINI)

    def test_with_string_name(self) -> None:
        """Gemini unittest: ToolCall with string name."""
        with Construct() as construct:
            llm.simulate(
                Provider.GEMINI,
                responses=[["Complete", ToolCall(name="search", arguments={"query": "info"})]],
            )
            tool.simulate(search, result=["info1"])

            gemini_search_agent.run("find info")

            tool.verify(search)
            self.assertEqual(construct.llm_calls[0].provider, "gemini")


# ============================================================================
# Simulating LLM Hallucinations and Invalid Tool Calls
# ============================================================================


@tenro.simulate
def test_llm_requests_nonexistent_tool() -> None:
    """Test agent handling of LLM requesting a tool that doesn't exist.

    This pattern is useful for testing defensive programming - ensuring your
    agent gracefully handles cases where the LLM hallucinates a tool name
    or requests a tool that isn't registered.
    """
    llm.simulate(
        Provider.OPENAI,
        responses=[
            [
                "I'll use the magic_tool to help",
                ToolCall(name="nonexistent_magic_tool", arguments={"spell": "abracadabra"}),
            ]
        ],
    )

    defensive_agent.run("Do something magical")

    llm.verify(Provider.OPENAI)

    llm_call = llm.calls()[0]
    assert llm_call.response is not None


@tenro.simulate
def test_llm_requests_tool_with_invalid_args() -> None:
    """Test agent handling of LLM providing invalid arguments to a tool.

    Another defensive programming pattern - the LLM might call a real tool
    but with arguments that don't match the expected schema.
    """
    llm.simulate(
        Provider.OPENAI,
        responses=[
            [
                "Searching with limit",
                ToolCall(name="search", arguments={"query": "test", "limit": "not_a_number"}),
            ]
        ],
    )
    tool.simulate(search, result=["result"])

    openai_search_agent.run("Search with bad args")

    llm.verify(Provider.OPENAI)


# ============================================================================
# LLMResponse with Blocks (Interleaving)
# ============================================================================
# Text blocks represent the model's reasoning BEFORE/WHILE making tool call
# requests. The model hasn't seen tool results yet - those come in a later turn.
#
# KEY DISTINCTION:
#   responses=[A, B, C]           → 3 separate LLM calls (3 turns)
#   responses=[LLMResponse([A, B, C])]  → 1 LLM call with interleaved content


@tenro.simulate
def test_single_turn_vs_multiple_turns() -> None:
    """Outer list = number of LLM calls. LLMResponse = content within one call.

    This is the key distinction: responses=[A, B] means TWO LLM calls,
    while responses=[LLMResponse([A, B])] means ONE call with both items.
    """
    # TWO separate LLM calls
    llm.simulate(Provider.OPENAI, responses=["First", "Second"])
    multi_turn_agent.run("topic")
    llm.verify_many(Provider.OPENAI, count=2)


@tenro.simulate
def test_anthropic_interleaved_single_turn() -> None:
    """Anthropic: ONE call with interleaved text + tool calls.

    Anthropic preserves block order. Text blocks are the model's reasoning
    as it decides which tools to call - all in a single atomic response.
    """
    llm.simulate(
        Provider.ANTHROPIC,
        responses=[
            LLMResponse(
                [
                    "I'll search for info.",
                    ToolCall(search, query="quantum"),
                    "Also checking weather.",
                    ToolCall(get_weather, city="NYC"),
                ]
            )
        ],
    )
    tool.simulate(search, result=["quantum info"])
    tool.simulate(get_weather, result={"temp": 72})

    anthropic_multi_tool_agent.run("Research")

    llm.verify_many(Provider.ANTHROPIC, count=1)  # ONE call
    tool.verify_many(count=2)


@tenro.simulate
def test_gemini_interleaved_single_turn() -> None:
    """Gemini: ONE call with interleaved content. Block order preserved."""
    llm.simulate(
        Provider.GEMINI,
        responses=[LLMResponse(["Searching now.", ToolCall(search, query="AI trends")])],
    )
    tool.simulate(search, result=["trend1"])

    gemini_search_agent.run("Find trends")

    llm.verify_many(Provider.GEMINI, count=1)
    tool.verify(search)


@tenro.simulate
def test_openai_blocks_flattened() -> None:
    """OpenAI: Blocks are flattened (no interleaving support).

    OpenAI Chat API concatenates text blocks and extracts tool calls
    to a separate array. Order within the turn is lost, but it still works.
    """
    llm.simulate(
        Provider.OPENAI,
        responses=[LLMResponse(["First.", ToolCall(search, query="test"), "Second."])],
    )
    tool.simulate(search, result=["result"])

    openai_search_agent.run("Search")

    llm.verify(Provider.OPENAI)
    tool.verify(search)


@tenro.simulate
def test_llmresponse_tool_calls_only() -> None:
    """LLMResponse with tool calls only (no text)."""
    llm.simulate(
        Provider.ANTHROPIC,
        responses=[LLMResponse([ToolCall(search, query="silent")])],
    )
    tool.simulate(search, result=["found"])

    anthropic_search_agent.run("Quick search")

    llm.verify(Provider.ANTHROPIC)
    tool.verify(search)


@tenro.simulate
def test_list_shorthand_equals_llmresponse() -> None:
    """List shorthand is equivalent to LLMResponse(blocks=[...])."""
    llm.simulate(
        Provider.ANTHROPIC,
        responses=[
            ["Reasoning", ToolCall(search, query="test")],  # shorthand
        ],
    )
    tool.simulate(search, result=["result"])

    anthropic_search_agent.run("Test")

    llm.verify(Provider.ANTHROPIC)
    tool.verify(search)
