# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pattern: Dynamic behavior with side_effect.

Shows how to make simulated responses depend on input arguments.
Uses realistic LLM-driven agent patterns.
"""

from __future__ import annotations

from examples.myapp import WeatherAgent, get_weather

import tenro
from tenro import Provider, ToolCall
from tenro.simulate import llm, tool


@tenro.simulate
def test_input_dependent_responses() -> None:
    """Responses vary based on input arguments using side_effect."""

    # side_effect: function receives the same args as the real function
    def weather_by_city(city: str) -> dict:
        weather_data = {
            "San Francisco": {"temp": 65, "condition": "foggy"},
            "Miami": {"temp": 85, "condition": "sunny"},
        }
        return weather_data.get(city, {"temp": 70, "condition": "unknown"})

    tool.simulate(get_weather, side_effect=weather_by_city)
    llm.simulate(
        Provider.OPENAI,
        responses=[
            [ToolCall("get_weather", city="San Francisco")],
            "San Francisco is 65°F and foggy.",
        ],
    )

    result = WeatherAgent().run("What's the weather in San Francisco?")

    assert result == "San Francisco is 65°F and foggy."
    tool.verify_many(get_weather, count=1)
    llm.verify_many(Provider.OPENAI, count=2)


@tenro.simulate
def test_side_effect_with_state() -> None:
    """Side effect can maintain state across multiple tool calls."""
    call_count = {"n": 0}

    def counting_weather(city: str) -> dict:
        call_count["n"] += 1
        return {"temp": 70 + call_count["n"], "call_number": call_count["n"]}

    tool.simulate(get_weather, side_effect=counting_weather)
    llm.simulate(
        Provider.OPENAI,
        responses=[
            [
                ToolCall("get_weather", city="NYC"),
                ToolCall("get_weather", city="LA"),
            ],
            "NYC is 71°F, LA is 72°F.",
        ],
    )

    result = WeatherAgent().run("Weather in NYC and LA?")

    assert result == "NYC is 71°F, LA is 72°F."
    tool.verify_many(get_weather, count=2)
    assert call_count["n"] == 2
