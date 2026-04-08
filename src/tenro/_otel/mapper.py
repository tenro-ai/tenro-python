# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Stateless semconv mapper: Tenro spans to OTel attribute dicts."""

from __future__ import annotations

from typing import Any

from tenro._core.spans import AgentRun, BaseSpan, LLMCall, ToolCall
from tenro._otel.constants import (
    _GENERATE_CONTENT_PROVIDERS,
    ATTR_AGENT_ID,
    ATTR_AGENT_NAME,
    ATTR_ERROR_TYPE,
    ATTR_EVAL_EXPLANATION,
    ATTR_EVAL_NAME,
    ATTR_EVAL_SCORE_LABEL,
    ATTR_EVAL_SCORE_VALUE,
    ATTR_INPUT_TOKENS,
    ATTR_OPERATION_NAME,
    ATTR_OUTPUT_TOKENS,
    ATTR_PROVIDER_NAME,
    ATTR_REQUEST_MODEL,
    ATTR_RESPONSE_FINISH_REASONS,
    ATTR_RESPONSE_ID,
    ATTR_RESPONSE_MODEL,
    ATTR_TOOL_CALL_ID,
    ATTR_TOOL_NAME,
    EVENT_EVALUATION_RESULT,
    OP_CHAT,
    OP_EXECUTE_TOOL,
    OP_GENERATE_CONTENT,
    OP_INVOKE_AGENT,
)
from tenro._otel.types import MappedEvent


def map_span_start(span: BaseSpan) -> dict[str, Any]:
    """Map span start fields to semconv attributes.

    Args:
        span: A Tenro span (LLMCall, ToolCall, or AgentRun).

    Returns:
        Dict of semconv attributes for span start.

    Raises:
        TypeError: If span is not a recognized subclass.
    """
    if isinstance(span, LLMCall):
        return _map_llm_start(span)
    if isinstance(span, ToolCall):
        return _map_tool_start(span)
    if isinstance(span, AgentRun):
        return _map_agent_start(span)
    raise TypeError(f"Unsupported span type: {type(span).__name__}")


def map_span_end(span: BaseSpan) -> dict[str, Any]:
    """Map span end fields to semconv attributes.

    Args:
        span: A Tenro span (LLMCall, ToolCall, or AgentRun).

    Returns:
        Dict of semconv attributes for span end.

    Raises:
        TypeError: If span is not a recognized subclass.
    """
    if isinstance(span, LLMCall):
        return _map_llm_end(span)
    if isinstance(span, (ToolCall, AgentRun)):
        attrs: dict[str, Any] = {}
        if span.status_code == "error" and span.error:
            attrs[ATTR_ERROR_TYPE] = span.error
        return attrs
    raise TypeError(f"Unsupported span type: {type(span).__name__}")


def get_otel_span_name(span: BaseSpan) -> str:
    """Derive semconv span name as "{operation} {identifier}".

    Args:
        span: A Tenro span.

    Returns:
        Span name string following GenAI semconv.

    Raises:
        TypeError: If span is not a recognized subclass.
    """
    op = get_operation_name(span)

    if isinstance(span, LLMCall):
        return f"{op} {span.model}" if span.model else op
    if isinstance(span, ToolCall):
        name = span.display_name or span.target_path
        return f"{op} {name}"
    if isinstance(span, AgentRun):
        return f"{op} {span.display_name}" if span.display_name else op
    raise TypeError(f"Unsupported span type: {type(span).__name__}")


def get_otel_span_kind(span: BaseSpan) -> str:
    """Read OTel span kind directly from span.

    Args:
        span: A Tenro span.

    Returns:
        "CLIENT" or "INTERNAL".
    """
    return span.kind


def get_operation_name(span: BaseSpan) -> str:
    """Derive gen_ai.operation.name from span type and provider.

    Args:
        span: A Tenro span.

    Returns:
        Operation name string.

    Raises:
        TypeError: If span is not a recognized subclass.
    """
    if isinstance(span, LLMCall):
        if span.provider in _GENERATE_CONTENT_PROVIDERS:
            return OP_GENERATE_CONTENT
        return OP_CHAT
    if isinstance(span, ToolCall):
        return OP_EXECUTE_TOOL
    if isinstance(span, AgentRun):
        return OP_INVOKE_AGENT
    raise TypeError(f"Unsupported span type: {type(span).__name__}")


def map_eval_result(
    *,
    eval_name: str,
    score: float,
    label: str | None = None,
    explanation: str | None = None,
    response_id: str | None = None,
) -> MappedEvent:
    """Map evaluation result to gen_ai.evaluation.result event.

    Args:
        eval_name: Name of the evaluation (e.g., "accuracy").
        score: Numeric score value.
        label: Optional human-readable score label.
        explanation: Optional explanation of the evaluation result.
        response_id: Optional gen_ai.response.id for correlation when
            span ID is unavailable.

    Returns:
        MappedEvent with event name and body dict.
    """
    body: dict[str, Any] = {
        ATTR_EVAL_NAME: eval_name,
        ATTR_EVAL_SCORE_VALUE: score,
    }
    if label is not None:
        body[ATTR_EVAL_SCORE_LABEL] = label
    if explanation is not None:
        body[ATTR_EVAL_EXPLANATION] = explanation
    if response_id is not None:
        body[ATTR_RESPONSE_ID] = response_id

    return MappedEvent(name=EVENT_EVALUATION_RESULT, body=body)


def _map_llm_start(span: LLMCall) -> dict[str, Any]:
    """Map LLMCall start attributes."""
    attrs: dict[str, Any] = {
        ATTR_OPERATION_NAME: get_operation_name(span),
        ATTR_PROVIDER_NAME: span.provider,
    }
    if span.model is not None:
        attrs[ATTR_REQUEST_MODEL] = span.model
    return attrs


def _map_llm_end(span: LLMCall) -> dict[str, Any]:
    """Map LLMCall end attributes."""
    attrs: dict[str, Any] = {}

    if span.token_usage is not None:
        input_tokens = span.token_usage.get("input", 0)
        # Include cache tokens in input total per semconv
        input_tokens += span.token_usage.get("cache_read", 0)
        input_tokens += span.token_usage.get("cache_creation", 0)
        attrs[ATTR_INPUT_TOKENS] = input_tokens
        attrs[ATTR_OUTPUT_TOKENS] = span.token_usage.get("output", 0)

    if span.response_model is not None:
        attrs[ATTR_RESPONSE_MODEL] = span.response_model
    if span.finish_reasons is not None:
        attrs[ATTR_RESPONSE_FINISH_REASONS] = span.finish_reasons
    if span.response_id is not None:
        attrs[ATTR_RESPONSE_ID] = span.response_id
    if span.status_code == "error" and span.error:
        attrs[ATTR_ERROR_TYPE] = span.error

    return attrs


def _map_tool_start(span: ToolCall) -> dict[str, Any]:
    """Map ToolCall start attributes."""
    attrs: dict[str, Any] = {
        ATTR_OPERATION_NAME: OP_EXECUTE_TOOL,
        ATTR_TOOL_NAME: span.display_name or span.target_path,
    }
    if span.tool_call_id is not None:
        attrs[ATTR_TOOL_CALL_ID] = span.tool_call_id
    return attrs


def _map_agent_start(span: AgentRun) -> dict[str, Any]:
    """Map AgentRun start attributes."""
    attrs: dict[str, Any] = {
        ATTR_OPERATION_NAME: OP_INVOKE_AGENT,
        ATTR_AGENT_ID: span.span_id,
    }
    if span.display_name is not None:
        attrs[ATTR_AGENT_NAME] = span.display_name
    return attrs
