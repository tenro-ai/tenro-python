# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pattern: Verifying tool and LLM calls.

Shows how to verify your tools and LLMs were called the expected number of times.
Uses realistic LLM-driven agent patterns.
"""

from __future__ import annotations

from examples.myapp import (
    DataAgent,
    WeatherAgent,
    get_weather,
    search_database,
)

from tenro import Provider, ToolCall
from tenro.simulate import llm, tool
from tenro.testing import tenro


@tenro
def test_exact_call_count() -> None:
    """Verify a tool was called exactly N times."""
    tool.simulate(get_weather, result={"temp": 72, "condition": "sunny"})
    llm.simulate(
        Provider.OPENAI,
        responses=[
            [ToolCall("get_weather", city="NYC")],
            "NYC is 72°F and sunny.",
        ],
    )

    WeatherAgent().run("What's the weather in NYC?")

    tool.verify_many(get_weather, count=1)
    llm.verify_many(Provider.OPENAI, count=2)


@tenro
def test_minimum_calls() -> None:
    """Verify at least N calls were made."""
    tool.simulate(get_weather, result={"temp": 70, "condition": "clear"})
    llm.simulate(
        Provider.OPENAI,
        responses=[
            [ToolCall("get_weather", city="A")],
            "Weather checked.",
        ],
    )

    WeatherAgent().run("Check weather")

    tool.verify_many(get_weather, at_least=1)
    llm.verify_many(Provider.OPENAI, at_least=1)


@tenro
def test_maximum_calls() -> None:
    """Verify no more than N calls were made."""
    tool.simulate(search_database, result=[{"id": 1}])
    llm.simulate(
        Provider.OPENAI,
        responses=[
            [ToolCall("search_database", query="test")],
            "Found 1 result.",
        ],
    )

    DataAgent().run("Search for test")

    tool.verify_many(search_database, at_most=5)
    llm.verify_many(Provider.OPENAI, at_most=5)


@tenro
def test_call_count_range() -> None:
    """Verify calls fall within a range."""
    tool.simulate(get_weather, result={"temp": 75})
    llm.simulate(
        Provider.OPENAI,
        responses=[
            [ToolCall("get_weather", city="LA")],
            "LA is 75°F.",
        ],
    )

    WeatherAgent().run("LA weather?")

    tool.verify_many(get_weather, at_least=1, at_most=5)
    llm.verify_many(Provider.OPENAI, at_least=1, at_most=5)


@tenro
def test_verify_at_least_once() -> None:
    """Verify a tool was called at least once (1 or more).

    Use verify() when you only care that something was called,
    not the exact number of times.
    """
    tool.simulate(get_weather, result={"temp": 72})
    llm.simulate(
        Provider.OPENAI,
        responses=[
            [ToolCall("get_weather", city="NYC")],
            "It's 72°F.",
        ],
    )

    WeatherAgent().run("NYC weather?")

    # verify() = at least once (passes if called 1, 2, 3, ... times)
    tool.verify(get_weather)
    llm.verify(Provider.OPENAI)


@tenro
def test_verify_exactly_once() -> None:
    """Verify a tool was called exactly once (not 0, not 2+).

    Use verify_many(count=1) when you need to assert the precise count.
    """
    tool.simulate(get_weather, result={"temp": 72})
    llm.simulate(
        Provider.OPENAI,
        responses=[
            [ToolCall("get_weather", city="NYC")],
            "It's 72°F.",
        ],
    )

    WeatherAgent().run("NYC weather?")

    # verify_many(count=1) = exactly once (fails if called 0 or 2+ times)
    tool.verify_many(get_weather, count=1)
    llm.verify_many(Provider.OPENAI, count=2)


@tenro
def test_verify_never_called() -> None:
    """Verify a tool was never called."""
    # LLM answers directly without using tools
    llm.simulate(Provider.OPENAI, response="I don't need to check the weather.")

    WeatherAgent().run("What color is the sky?")

    tool.verify_never(get_weather)
    llm.verify(Provider.OPENAI)
