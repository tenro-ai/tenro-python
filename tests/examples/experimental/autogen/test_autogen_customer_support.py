# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Customer Support: Testing knowledge base retrieval with AutoGen."""

from examples.experimental.autogen.myapp.agents import CustomerSupportAgent, search_kb

from tenro import Provider
from tenro.simulate import agent, llm, tool
from tenro.tool_calls import tc


def test_customer_support_answers_question(construct) -> None:
    """Test customer support agent uses knowledge base and LLM.

    Flow: LLM decides to call search_kb → tool returns result → LLM answers.
    """
    # Simulate the tool that LLM will call
    tool.simulate(search_kb, result="Full refunds within 30 days.")

    # Simulate LLM: first requests tool, then gives final answer
    llm.simulate(
        Provider.OPENAI,
        responses=[
            {"text": "", "tool_calls": [tc("search_kb", query="refund")]},
            {"text": "You can get a full refund within 30 days of purchase. TERMINATE"},
        ],
    )

    result = CustomerSupportAgent().run("How do I get a refund?")

    assert result == "You can get a full refund within 30 days of purchase. TERMINATE"
    agent.verify(CustomerSupportAgent)
    llm.verify_many(Provider.OPENAI, count=2)
    tool.verify_many(search_kb, count=1)
