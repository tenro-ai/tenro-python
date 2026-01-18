# LangChain Examples

Testing LangChain chains and agents with Tenro.

## Requirements

```bash
# Set API key (required for import, but calls are intercepted by Tenro)
export OPENAI_API_KEY=sk-your-key-or-placeholder

# Install dependencies
uv sync --group examples
```

## Run

```bash
uv run pytest examples/langchain/ -v
```

## Examples

| File | Description |
|------|-------------|
| `test_langchain_customer_support.py` | Knowledge base search + LLM response |
| `test_langchain_multi_turn_conversation.py` | Multi-turn chat with memory |
| `test_langchain_rag_pipeline.py` | RAG with document retrieval |
| `test_langchain_thirdparty_tools.py` | Testing with third-party tools |
