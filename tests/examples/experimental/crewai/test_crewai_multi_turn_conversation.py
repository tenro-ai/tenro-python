# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Multi-Turn Conversation: Testing sequential LLM calls with CrewAI.

NOTE: CrewAI uses ReAct text parsing, not OpenAI's native `tool_calls` structure.
Unlike LangChain/OpenAI SDK, CrewAI expects "Final Answer:" in the LLM's text output.
Simulated responses must match this format.
"""

from __future__ import annotations

from tenro import Provider
from tenro.simulate import llm
from tenro.testing import tenro


def react_final(answer: str) -> str:
    """Format a ReAct final answer that CrewAI's parser understands."""
    return f"Thought: I know the answer.\nFinal Answer: {answer}\n"


@tenro
def test_simple_crewai_task() -> None:
    """Test a simple CrewAI task without memory (no chromadb)."""
    from crewai import Agent, Crew, Task

    llm.simulate(
        Provider.OPENAI,
        responses=[react_final("Python lists are created with square brackets: [1, 2, 3]")],
    )

    # memory=False: CrewAI's memory triggers chromadb which makes additional
    # unpredictable LLM calls for embeddings, breaking deterministic simulation
    agent = Agent(
        role="Coding Assistant",
        goal="Help with Python questions",
        backstory="You are a Python expert.",
        llm="gpt-4o-mini",
        memory=False,
    )

    task = Task(
        description="How do I create a list in Python?",
        expected_output="A helpful response about Python lists",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], memory=False)
    result = crew.kickoff()

    assert str(result) == "Python lists are created with square brackets: [1, 2, 3]"
    # No agent.verify() - this test uses inline CrewAI Agent, not @link_agent decorated class
    llm.verify(Provider.OPENAI)
