# Pydantic AI Examples

Testing Pydantic AI agents with Tenro.

## Requirements

```bash
# Set API key (required for import, but calls are intercepted by Tenro)
export OPENAI_API_KEY=sk-your-key-or-placeholder

# Install dependencies
uv sync --group examples
```

## Run

```bash
uv run pytest examples/pydantic_ai/ -v
```

## Examples

| File | Description |
|------|-------------|
| `test_pydantic_ai_customer_support.py` | Agent with tool calling |
| `test_pydantic_ai_multi_turn_conversation.py` | Multi-turn agent conversation |
| `test_pydantic_ai_rag_pipeline.py` | RAG with structured outputs |
