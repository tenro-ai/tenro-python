# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry GenAI semantic convention constants.

Pinned to semconv 1.40.0. Any future semconv upgrade must be treated
as an explicit compatibility change in the mapper layer.
"""

# Semconv version and schema URL
SEMCONV_VERSION = "1.40.0"
SCHEMA_URL = "https://opentelemetry.io/schemas/1.40.0"

# Core attributes
ATTR_OPERATION_NAME = "gen_ai.operation.name"
ATTR_PROVIDER_NAME = "gen_ai.provider.name"
ATTR_REQUEST_MODEL = "gen_ai.request.model"
ATTR_RESPONSE_MODEL = "gen_ai.response.model"
ATTR_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ATTR_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
ATTR_RESPONSE_ID = "gen_ai.response.id"
ATTR_OUTPUT_TYPE = "gen_ai.output.type"
ATTR_AGENT_NAME = "gen_ai.agent.name"
ATTR_AGENT_ID = "gen_ai.agent.id"
ATTR_TOOL_NAME = "gen_ai.tool.name"
ATTR_TOOL_CALL_ID = "gen_ai.tool.call.id"
ATTR_SERVER_ADDRESS = "server.address"
ATTR_SERVER_PORT = "server.port"
ATTR_ERROR_TYPE = "error.type"

# Optional content attributes
ATTR_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
ATTR_INPUT_MESSAGES = "gen_ai.input.messages"
ATTR_OUTPUT_MESSAGES = "gen_ai.output.messages"

# Evaluation attributes
ATTR_EVAL_NAME = "gen_ai.evaluation.name"
ATTR_EVAL_SCORE_VALUE = "gen_ai.evaluation.score.value"
ATTR_EVAL_SCORE_LABEL = "gen_ai.evaluation.score.label"
ATTR_EVAL_EXPLANATION = "gen_ai.evaluation.explanation"

# Event names
EVENT_INFERENCE_DETAILS = "gen_ai.client.inference.operation.details"
EVENT_EVALUATION_RESULT = "gen_ai.evaluation.result"

# Operation names
OP_CHAT = "chat"
OP_TEXT_COMPLETION = "text_completion"
OP_GENERATE_CONTENT = "generate_content"
OP_EMBEDDINGS = "embeddings"
OP_EXECUTE_TOOL = "execute_tool"
OP_INVOKE_AGENT = "invoke_agent"
OP_CREATE_AGENT = "create_agent"
OP_RETRIEVAL = "retrieval"

# Providers that use generate_content operation name
_GENERATE_CONTENT_PROVIDERS = frozenset({"gemini", "google", "vertex_ai"})
