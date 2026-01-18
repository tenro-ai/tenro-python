# LangGraph Examples

Testing LangGraph stateful workflows with Tenro.

## Requirements

```bash
# Set API key (required for import, but calls are intercepted by Tenro)
export OPENAI_API_KEY=sk-your-key-or-placeholder

# Install dependencies
uv sync --group examples
```

## Run

```bash
uv run pytest examples/langgraph/ -v
```

## Examples

| File | Description |
|------|-------------|
| `test_langgraph_customer_support.py` | Stateful support workflow |
| `test_langgraph_multi_turn_conversation.py` | Multi-step conversation graph |
| `test_langgraph_rag_pipeline.py` | RAG workflow with state |
