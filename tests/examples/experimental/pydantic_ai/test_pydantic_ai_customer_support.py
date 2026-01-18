# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Customer Support: Testing knowledge base retrieval with Pydantic AI."""

from examples.experimental.pydantic_ai.myapp.agents import CustomerSupportAgent
from examples.myapp import search_knowledge_base

from tenro import Provider
from tenro.simulate import agent, llm, tool
from tenro.tool_calls import tc


def test_customer_support_answers_question(construct) -> None:
    """Test customer support agent uses knowledge base and LLM."""
    tool.simulate(
        search_knowledge_base,
        result=[{"title": "Refund Policy", "content": "Full refunds within 30 days."}],
    )
    # Simulate LLM requesting the tool, then returning final answer
    llm.simulate(
        Provider.OPENAI,
        responses=[
            {"text": "", "tool_calls": [tc("search_kb")]},
            {"text": "You can get a full refund within 30 days of purchase."},
        ],
    )

    result = CustomerSupportAgent().run("How do I get a refund?")

    assert result == "You can get a full refund within 30 days of purchase."
    agent.verify(CustomerSupportAgent)
    llm.verify_many(Provider.OPENAI, count=2)
    tool.verify_many(search_knowledge_base, count=1)
