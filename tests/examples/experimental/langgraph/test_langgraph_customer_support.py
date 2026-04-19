# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Customer Support: Testing knowledge base retrieval with LangGraph."""

from __future__ import annotations

from examples.experimental.langgraph.myapp.agents import CustomerSupportAgent
from examples.myapp import search_knowledge_base

import tenro
from tenro import Provider
from tenro.simulate import agent, llm, tool


@tenro.simulate
def test_customer_support_answers_question() -> None:
    """Test customer support agent uses knowledge base and LLM."""
    tool.simulate(
        search_knowledge_base,
        result=[{"title": "Refund Policy", "content": "Full refunds within 30 days."}],
    )
    llm.simulate(
        Provider.OPENAI,
        response="You can get a full refund within 30 days of purchase.",
    )

    result = CustomerSupportAgent().run("How do I get a refund?")

    assert result == "You can get a full refund within 30 days of purchase."
    agent.verify(CustomerSupportAgent)
    llm.verify(Provider.OPENAI)
    tool.verify_many(search_knowledge_base, count=1)
