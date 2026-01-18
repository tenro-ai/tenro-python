# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Customer Support: Testing knowledge base retrieval with CrewAI.

NOTE: CrewAI uses ReAct text parsing, not OpenAI's native `tool_calls` structure.
Unlike LangChain/OpenAI SDK, CrewAI expects "Action:" and "Action Input:" in the
LLM's text output. Simulated responses must match this format.
"""

import json

from examples.experimental.crewai.myapp.agents import CustomerSupportAgent, search_kb

from tenro import Provider
from tenro.simulate import agent, llm, tool


def react_action(tool_name: str, tool_input: dict) -> str:
    """Format a ReAct action that CrewAI's parser understands."""
    return (
        "Thought: I should search the knowledge base for refund information.\n"
        f"Action: {tool_name}\n"
        f"Action Input: {json.dumps(tool_input)}\n"
    )


def react_final(answer: str) -> str:
    """Format a ReAct final answer that CrewAI's parser understands."""
    return f"Thought: I now know the final answer.\nFinal Answer: {answer}\n"


def test_customer_support_answers_question(construct) -> None:
    """Test customer support agent uses knowledge base and LLM."""
    tool.simulate(search_kb.func, result="Full refunds within 30 days.")

    llm.simulate(
        Provider.OPENAI,
        responses=[
            react_action("Search Knowledge Base", {"query": "refund"}),
            react_final("You can get a full refund within 30 days of purchase."),
        ],
    )

    result = CustomerSupportAgent().run("How do I get a refund?")

    assert result == "You can get a full refund within 30 days of purchase."
    agent.verify(CustomerSupportAgent)
    llm.verify_many(Provider.OPENAI, count=2)
    tool.verify_many(search_kb.func, count=1)
