# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Verification API for construct testing.

Provides clean, expressive verifications for testing agent behavior:
- construct.verify_tool(fetch_data)  # at least once (1+)
- construct.verify_tools(count=1)  # exactly once
- construct.verify_tools(count=2)  # exactly twice
- construct.verify_llms(Provider.OPENAI, count=2)
- construct.verify_agent(risk_agent, threshold=0.8)
- construct.verify_llm(output_contains="success")
"""

from tenro._construct.verify.agent import AgentVerifications
from tenro._construct.verify.llm import LLMVerifications
from tenro._construct.verify.llm_scope import LLMScopeVerifications
from tenro._construct.verify.tool import ToolVerifications

__all__ = [
    "AgentVerifications",
    "LLMScopeVerifications",
    "LLMVerifications",
    "ToolVerifications",
]
