# CrewAI Examples

Testing CrewAI multi-agent crews with Tenro.

## Requirements

```bash
# Set API key (required for import, but calls are intercepted by Tenro)
export OPENAI_API_KEY=sk-your-key-or-placeholder

# Install dependencies
uv sync --group examples
```

## Run

```bash
uv run pytest examples/crewai/ -v
```

## Python Version

CrewAI requires Python 3.13 or lower (not compatible with 3.14).

## Examples

| File | Description |
|------|-------------|
| `test_crewai_customer_support.py` | Single agent crew |
| `test_crewai_multi_turn_conversation.py` | Multi-agent collaboration |
| `test_crewai_rag_pipeline.py` | RAG crew with researcher + writer |
