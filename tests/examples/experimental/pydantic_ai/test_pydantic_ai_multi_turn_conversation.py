# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Multi-Turn Conversation: Testing sequential LLM calls with Pydantic AI."""

from __future__ import annotations

from examples.experimental.pydantic_ai.myapp.agents import ConversationAgent

import tenro
from tenro import Provider
from tenro.simulate import agent, llm


@tenro.simulate
def test_multi_turn_conversation() -> None:
    """Test agent handles multi-turn conversation with context."""
    llm.simulate(
        Provider.OPENAI,
        responses=[
            "A list in Python is created with square brackets: my_list = [1, 2, 3]",
            "To add items, use append(): my_list.append(4)",
        ],
    )

    responses = ConversationAgent().run(
        ["How do I create a list in Python?", "How do I add items to it?"]
    )

    assert responses == [
        "A list in Python is created with square brackets: my_list = [1, 2, 3]",
        "To add items, use append(): my_list.append(4)",
    ]
    agent.verify(ConversationAgent)
    llm.verify_many(Provider.OPENAI, count=2)
