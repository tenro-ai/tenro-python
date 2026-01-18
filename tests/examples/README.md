# Tenro Examples

Runnable examples showing how to test AI agents with Tenro.

## API Keys

> ⚠️ Some frameworks validate API keys at import. Set the key for your provider before running:
> `export OPENAI_API_KEY={YOUR_KEY}` or `export ANTHROPIC_API_KEY={YOUR_KEY}`
>
> Tests use simulated responses—any non-empty value works. Never commit real keys.

## Running Examples

**SDK users** (pip install tenro):
```bash
# Install your framework
pip install tenro crewai  # or langchain, pydantic-ai, etc.

# Copy an example and run
python examples/experimental/crewai/test_crewai_customer_support.py
```

**Development** (this repo):
```bash
uv sync --group examples
uv run pytest examples/
```

## Structure

```
examples/
├── shared/              # Shared tools and clients (DRY)
│   ├── tools.py         # Business logic tools (search_knowledge_base, etc.)
│   └── clients.py       # LLM client factories
├── experimental/        # Best-effort, may break with framework updates
│   ├── autogen/myapp/
│   ├── crewai/myapp/
│   ├── langchain/myapp/
│   ├── langgraph/myapp/
│   ├── llamaindex/myapp/
│   └── pydantic_ai/myapp/
├── patterns/myapp/      # Tenro API patterns (direct SDK, no framework)
└── custom/myapp/        # Raw agent examples (direct SDK)
```

### Support Levels

| Directory | Support Level | CI Coverage |
|-----------|---------------|-------------|
| `experimental/` | **Best-effort** | Common scenarios |
| `patterns/` | **Maintained** | Full tests |
| `custom/` | **Maintained** | Full tests |

---

## experimental/

Third-party framework integrations. These are best-effort and may break with framework updates.

| Section | Purpose | User Question |
|---------|---------|---------------|
| **langchain/** | LangChain chains/agents | "How do I test my LangChain agent?" |
| **langgraph/** | LangGraph stateful workflows | "How do I test my LangGraph?" |
| **pydantic_ai/** | Pydantic AI agents | "How do I test Pydantic AI?" |
| **crewai/** | CrewAI multi-agent crews | "How do I test my CrewAI crew?" |
| **autogen/** | AutoGen conversations | "How do I test AutoGen agents?" |
| **llamaindex/** | LlamaIndex RAG pipelines | "How do I test LlamaIndex?" |

---

## patterns/

API feature demonstrations - learn what Tenro can do.

- **test_simulating_responses.py** — Control tool/LLM returns with `result=`, `results=[]`, `responses=[]`
- **test_simulating_errors.py** — Test error handling with exceptions in results
- **test_simulating_tool_calls.py** — Simulate LLM tool calls with `tc()` helper and `tool_calls=`
- **test_verifying_calls.py** — Assert call counts with `verify_many(count=, at_least=, at_most=)`
- **test_verifying_content.py** — Check LLM responses with `output_contains=`, `call_index=`
- **test_verifying_never_called.py** — Ensure operations didn't happen with `verify_never()`
- **test_verifying_call_sequence.py** — Verify execution order with `verify_sequence()`
- **test_optional_simulations.py** — Handle conditional branches with `optional=True`
- **test_dynamic_behavior.py** — Input-dependent responses with `side_effect=`
- **test_default_provider.py** — Set default provider to omit `provider=` argument

---

## custom/

Raw agent examples without frameworks - for custom agent implementations.

- **test_custom_customer_support.py** — Knowledge base search + LLM response
- **test_custom_rag_pipeline.py** — Document retrieval + synthesis
- **test_custom_multi_turn_conversation.py** — Multi-turn conversation with history
- **test_email_summarizer_agent.py** — Fetch emails + summarize
- **test_research_assistant_agent.py** — Web search + synthesize findings
- **test_sales_outreach_agent.py** — CRM lookup + pricing + generate pitch
- **test_code_review_agent.py** — Fetch PR diff + analyze + post comment
- **test_meeting_notes_agent.py** — Transcribe + extract action items
- **test_voice_call_agent.py** — Speech-to-text + LLM + text-to-speech
- **test_rag_document_agent.py** — Vector search + retrieval + summarize
- **test_recruitment_agent.py** — Screen candidates + schedule interviews
