# AutoGen Examples

Testing AutoGen conversational agents with Tenro.

## Requirements

```bash
# Set API key (required for import, but calls are intercepted by Tenro)
export OPENAI_API_KEY=sk-your-key-or-placeholder

# Install dependencies
uv sync --group examples
```

## Run

```bash
uv run pytest examples/autogen/ -v
```

## Python Version

AutoGen requires Python 3.13 or lower (not compatible with 3.14).

## Examples

| File | Description |
|------|-------------|
| `test_autogen_customer_support.py` | Basic assistant agent |
| `test_autogen_multi_turn_conversation.py` | Multi-turn conversation |
| `test_autogen_rag_pipeline.py` | RAG with document retrieval |
