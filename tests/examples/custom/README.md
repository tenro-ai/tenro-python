# Custom Agent Examples

Testing raw agent implementations without frameworks.

## Requirements

```bash
# Set API key (required for import, but calls are intercepted by Tenro)
export OPENAI_API_KEY=sk-your-key-or-placeholder

# Install dependencies
uv sync --group examples
```

## Run

```bash
uv run pytest examples/custom/ -v
```

## Examples

| File | Description |
|------|-------------|
| `test_custom_customer_support.py` | Knowledge base search + LLM |
| `test_custom_multi_turn_conversation.py` | Multi-turn chat agent |
| `test_custom_rag_pipeline.py` | RAG with vector search |
| `test_email_summarizer_agent.py` | Fetch emails + summarize |
| `test_research_assistant_agent.py` | Web search + synthesize |
| `test_sales_outreach_agent.py` | CRM lookup + generate pitch |
| `test_code_review_agent.py` | PR diff analysis |
| `test_meeting_notes_agent.py` | Transcribe + extract actions |
| `test_voice_call_agent.py` | Speech-to-text + LLM + TTS |
| `test_rag_document_agent.py` | Vector search + summarize |
| `test_recruitment_agent.py` | Screen candidates + schedule |
