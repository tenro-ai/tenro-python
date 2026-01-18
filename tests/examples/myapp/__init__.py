# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Shared application code for example tests.

This module contains all shared business logic, LLM clients, tools, and agents
that are used across the custom and patterns example tests. Framework-specific
examples (experimental/) have their own myapp/ directories.

Contents:
- tools.py: Business logic tools (search_knowledge_base, get_weather, etc.)
- clients.py: LLM client factories (get_openai_client, get_anthropic_client)
- agents.py: LLM wrappers and agent classes
"""

from __future__ import annotations

# LLM wrappers and agents
from examples.myapp.agents import (
    ConversationAgent,
    CustomerSupportAgent,
    DataAgent,
    DefensiveAgent,
    MultiToolAgent,
    MultiTurnAgent,
    PersistenceAgent,
    PipelineAgent,
    ProcessingAgent,
    ProviderSearchAgent,
    RAGPipeline,
    ResilientAgent,
    SafeCleanupAgent,
    SearchAgent,
    SmartCacheAgent,
    TopicConversationAgent,
    ValidationAgent,
    WeatherAgent,
    anthropic_multi_tool_agent,
    anthropic_search_agent,
    call_anthropic,
    call_gemini,
    call_llm,
    call_openai,
    chat,
    chat_with_tools,
    defensive_agent,
    gemini_multi_tool_agent,
    gemini_search_agent,
    generate_response,
    get_data_with_cache,
    multi_step_workflow,
    multi_turn_agent,
    openai_multi_tool_agent,
    openai_search_agent,
)

# Clients
from examples.myapp.clients import (
    get_anthropic_client,
    get_openai_client,
)

# Tools
from examples.myapp.tools import (
    call_api,
    check_cache,
    delete_all_records,
    fetch_documents,
    fetch_emails,
    fetch_from_api,
    fetch_weather,
    get_cached_data,
    get_weather,
    process_data,
    save_result,
    search,
    search_database,
    search_documents,
    search_knowledge_base,
    send_email,
    summarize_email,
    validate_input,
)

__all__ = [
    # Agent classes
    "ConversationAgent",
    "CustomerSupportAgent",
    "DataAgent",
    "DefensiveAgent",
    "MultiToolAgent",
    "MultiTurnAgent",
    "PersistenceAgent",
    "PipelineAgent",
    "ProcessingAgent",
    "ProviderSearchAgent",
    "RAGPipeline",
    "ResilientAgent",
    "SafeCleanupAgent",
    "SearchAgent",
    "SmartCacheAgent",
    "TopicConversationAgent",
    "ValidationAgent",
    "WeatherAgent",
    # Agent instances
    "anthropic_multi_tool_agent",
    "anthropic_search_agent",
    # LLM wrappers
    "call_anthropic",
    # Tools
    "call_api",
    "call_gemini",
    "call_llm",
    "call_openai",
    "chat",
    "chat_with_tools",
    "check_cache",
    "defensive_agent",
    "delete_all_records",
    "fetch_documents",
    "fetch_emails",
    "fetch_from_api",
    "fetch_weather",
    "gemini_multi_tool_agent",
    "gemini_search_agent",
    "generate_response",
    # Clients
    "get_anthropic_client",
    "get_cached_data",
    # Helper functions
    "get_data_with_cache",
    "get_openai_client",
    "get_weather",
    "multi_step_workflow",
    "multi_turn_agent",
    "openai_multi_tool_agent",
    "openai_search_agent",
    "process_data",
    "save_result",
    "search",
    "search_database",
    "search_documents",
    "search_knowledge_base",
    "send_email",
    "summarize_email",
    "validate_input",
]
