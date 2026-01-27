# Tenro

[![PyPI version](https://img.shields.io/pypi/v/tenro.svg)](https://pypi.org/project/tenro/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/tenro/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

`Tenro` is a modern **simulation framework for testing AI agents**. Simulate multi-agent workflows and tool usage **without burning tokens**.

- **No API costs** — Tests run offline (no LLM calls)
- **Deterministic** — Simulate responses, errors, and tool results
- **Workflow verification** — Check tools, edge cases, and agent behaviours

## Install

```bash
pip install tenro
# or
uv add tenro
```

## Quick Start

Tenro provides a `construct` pytest fixture that intercepts LLM and tool calls during tests.

```python
# myapp/agent.py
from tenro import link_agent, link_tool

@link_tool
def search(query: str) -> list[str]:
    ...  # calls external API

@link_agent("Assistant", entry_points="run")
class AssistantAgent:
    def run(self, task: str) -> str:
        ...  # agent loop: LLM calls tools, returns final answer
```

```python
# tests/test_agent.py
from tenro import Provider, ToolCall
from tenro.simulate import llm, tool
from myapp.agent import search, AssistantAgent
from tenro.testing import tenro

@tenro
def test_agent():
    tool.simulate(search, result=["Simulated Doc"])
    llm.simulate(
        Provider.ANTHROPIC,
        responses=[
            ToolCall(search, query="Find docs"),
            "Summary of docs.",
        ],
    )

    result = AssistantAgent().run("Find docs")

    assert result == "Summary of docs."
    tool.verify(search)
    llm.verify_many(Provider.ANTHROPIC, count=2)
```

No mocks to configure, no expensive API calls, no flaky tests.

## Before / After

**Without Tenro** — manual mocks, helper functions, boilerplate:

```python
# test_helpers.py - you write and maintain this
def mock_llm_response(content=None, tool_call=None):
    if tool_call:
        message = ChatCompletionMessage(
            role="assistant", content=None,
            tool_calls=[ChatCompletionMessageToolCall(
                id="call_abc", type="function",
                function=Function(name=tool_call["name"], arguments=json.dumps(tool_call["args"]))
            )]
        )
    else:
        message = ChatCompletionMessage(role="assistant", content=content, tool_calls=None)
    return ChatCompletion(
        id="chatcmpl-123", created=0, model="gpt-5", object="chat.completion",
        choices=[Choice(index=0, finish_reason="stop", message=message)]
    )

# test_agent.py
@patch("myapp.tools.get_weather")
@patch("openai.chat.completions.create")
def test_agent(mock_llm, mock_weather):
    mock_weather.return_value = {"temp": 72, "condition": "sunny"}
    mock_llm.side_effect = [
        mock_llm_response(tool_call={"name": "get_weather", "args": {"city": "Paris"}}),
        mock_llm_response(content="It's 72°F and sunny in Paris."),
    ]
    result = my_agent.run("Weather in Paris?")
    assert result == "It's 72°F and sunny in Paris."
    mock_weather.assert_called_once_with(city="Paris")
```

**With Tenro:**

```python
from tenro import Provider, ToolCall
from tenro.simulate import llm, tool
from myapp.agent import get_weather, WeatherAgent
from tenro.testing import tenro

@tenro
def test_agent():
    tool.simulate(get_weather, result={"temp": 72, "condition": "sunny"})
    llm.simulate(
        Provider.OPENAI,
        responses=[
            ToolCall(get_weather, city="Paris"),
            "It's 72°F and sunny in Paris.",
        ],
    )

    result = WeatherAgent().run("Weather in Paris?")

    tool.verify(get_weather)
    llm.verify_many(Provider.OPENAI, count=2)
    assert result == "It's 72°F and sunny in Paris."
```

No patch decorators. No response builders. Just simulate and verify.

## How It Works

Tenro's `Construct` is a simulation environment for your AI agents. Link your functions with decorators, then test with full control:

```python
from tenro import link_agent, link_tool

@link_tool
def search(query: str) -> list[str]:
    ...  # calls external API

@link_agent("Manager", entry_points="run")
class ManagerAgent:
    def run(self, task: str) -> str:
        ...  # LLM calls search tool, summarizes results
```

During tests, `Construct` intercepts linked LLM and tool calls and returns your simulated results instead of calling the real provider.

## Simulation API

```python
from tenro import Provider
from tenro.simulate import llm, tool
from tenro import ToolCall
from myapp.agent import search, MyAgent
from tenro.testing import tenro

@tenro
def test_verification():
    # Setup
    tool.simulate(search, result=["doc1", "doc2"])
    llm.simulate(
        Provider.ANTHROPIC,
        responses=[
            ToolCall(search, query="docs"),
            "Summary",
        ],
    )

    # Run
    MyAgent().run("query")

    # Verify
    tool.verify(search)                              # at least once
    tool.verify_many(search, count=1)                # exactly once
    llm.verify_many(Provider.ANTHROPIC, count=2)     # exactly twice

    # Access call data
    assert llm.calls()[1].response == "Summary"
```

## Trace Output

Enable trace visualization to debug agent execution:

> Set `TENRO_TRACE=true` in your `.env` or run `TENRO_TRACE=true pytest`

```
🤖 SupportAgent
   ├─ → user: "My order #12345 hasn't arrived"
   │
   ├─ 🧠 claude-sonnet-4-5
   │     ├─ → prompt: "Help customer: My order #12345 hasn't arrived"
   │     └─ ← tool_call: lookup_order(order_id='12345')
   │
   ├─ 🔧 lookup_order
   │     ├─ → order_id='12345'
   │     └─ ← {'status': 'shipped', 'eta': '2025-01-02'}
   │
   ├─ 🧠 claude-sonnet-4-5
   │     ├─ → prompt: "Tool result: {'status': 'shipped', ...}"
   │     └─ ← "Your order has shipped and will arrive by Jan 2nd!"
   │
   └─ ← "Your order has shipped and will arrive by Jan 2nd!"

────────────────────────────────────────────────────────────────
Summary: 1 agent | 2 LLM calls | 1 tool call | Total: 1.24s
```

## LLM Provider Support

| Provider | Status |
|----------|:------:|
| OpenAI | ✅ |
| Anthropic | ✅ |
| Gemini | ✅ |
| Custom | Experimental |

## Compatibility

- Python 3.11+
- pytest 7.0+

## Contributing

Thanks for your interest in contributing!

Tenro is still in the early stages, focused on stabilizing the core API.
Pull requests are **not being accepted** at this time.

You can still help by:

- **⭐ Star the repo** to follow progress and help others discover it
- **Report bugs** (include repro steps + logs if possible)
- **Request features** (share the use case and expected behavior)
- **Ask questions** (usage, roadmap, design decisions)

Please use [GitHub Issues](https://github.com/tenro-ai/tenro-python/issues) for discussions and reports.

## License

[Apache 2.0](LICENSE)

## Support

- Issues: [GitHub Issues](https://github.com/tenro-ai/tenro-python/issues)
- Email: support@tenro.ai
