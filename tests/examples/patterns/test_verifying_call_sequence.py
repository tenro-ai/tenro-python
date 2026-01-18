# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pattern: Verifying call sequences.

Shows how to ensure tools and agents are called in the correct order.

Note: This demonstrates a pipeline/orchestration pattern where agents
coordinate in a fixed sequence. This is different from LLM-driven tool
calling - it tests that pipeline stages execute in order.
"""

from tenro import link_agent, link_tool
from tenro.simulate import agent, tool


# Tools for the pipeline
@link_tool
def validate_input(data: dict) -> bool:
    """Validate input before processing."""
    return True


@link_tool
def process_data(data: dict) -> dict:
    """Process the validated data."""
    return {"processed": True}


@link_tool
def save_result(result: dict) -> str:
    """Save the processed result."""
    return "saved"


# Pipeline agents (orchestration pattern)
@link_agent
class ValidationAgent:
    """Validates input data."""

    def run(self, data: dict) -> bool:
        return validate_input(data)


@link_agent
class ProcessingAgent:
    """Processes validated data."""

    def run(self, data: dict) -> dict:
        return process_data(data)


@link_agent
class PersistenceAgent:
    """Saves processed results."""

    def run(self, result: dict) -> str:
        return save_result(result)


@link_agent
class PipelineAgent:
    """Orchestrates the full pipeline: validate → process → save."""

    def run(self, data: dict) -> str:
        ValidationAgent().run(data)
        result = ProcessingAgent().run(data)
        return PersistenceAgent().run(result)


def test_pipeline_tools_execute_in_order(construct) -> None:
    """Verify pipeline tools are called in sequence."""
    tool.simulate(validate_input, result=True)
    tool.simulate(process_data, result={"processed": True})
    tool.simulate(save_result, result="saved")

    result = PipelineAgent().run({"input": "data"})

    assert result == "saved"
    tool.verify_many(validate_input, count=1)
    tool.verify_many(process_data, count=1)
    tool.verify_many(save_result, count=1)


def test_pipeline_agents_called(construct) -> None:
    """Verify orchestrator delegates to sub-agents."""
    tool.simulate(validate_input, result=True)
    tool.simulate(process_data, result={"processed": True})
    tool.simulate(save_result, result="saved")

    PipelineAgent().run({"input": "data"})

    agent.verify(PipelineAgent)
    agent.verify(ValidationAgent)
    agent.verify(ProcessingAgent)
    agent.verify(PersistenceAgent)
