# LlamaIndex Examples

Testing LlamaIndex RAG pipelines with Tenro.

## Requirements

```bash
# Set API key (required for import, but calls are intercepted by Tenro)
export OPENAI_API_KEY=sk-your-key-or-placeholder

# Install dependencies
uv sync --group examples
```

## Run

```bash
uv run pytest examples/llamaindex/ -v
```

## Python Version

LlamaIndex requires Python 3.13 or lower (not compatible with 3.14).

## Examples

| File | Description |
|------|-------------|
| `test_llamaindex_customer_support.py` | Basic query engine |
| `test_llamaindex_multi_turn_conversation.py` | Chat with memory |
| `test_llamaindex_rag_pipeline.py` | Full RAG pipeline |
