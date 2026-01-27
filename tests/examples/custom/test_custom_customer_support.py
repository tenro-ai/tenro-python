# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Customer Support: Testing knowledge base retrieval with custom OpenAI agents."""

from __future__ import annotations

from examples.myapp import CustomerSupportAgent, generate_response, search_knowledge_base

from tenro import Provider
from tenro.simulate import llm, tool
from tenro.testing import tenro


@tenro
def test_customer_support_answers_question() -> None:
    """Test customer support agent uses knowledge base and LLM."""
    tool.simulate(
        search_knowledge_base,
        result=[{"title": "Refund Policy", "content": "Full refunds within 30 days."}],
    )
    llm.simulate(
        Provider.OPENAI,
        target=generate_response,
        response="You can get a full refund within 30 days of purchase.",
    )

    CustomerSupportAgent().run("How do I get a refund?")

    tool.verify_many(search_knowledge_base, count=1)
    llm.verify(Provider.OPENAI)
    llm.verify(output_contains="refund")
