# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Multi-Turn Conversation: Testing sequential LLM calls with custom OpenAI agents."""

from __future__ import annotations

from examples.myapp import ConversationAgent

import tenro
from tenro import Provider
from tenro.simulate import llm


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

    assert len(responses) == 2
    llm.verify_many(Provider.OPENAI, count=2)
    llm.verify(output_contains="list", call_index=0)
    llm.verify(output_contains="append", call_index=1)
