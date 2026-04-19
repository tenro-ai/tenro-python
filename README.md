<div align="center">

# Tenro — Simulation Harness for Testing AI Agents

Simulate agent workflows and verify behavior without burning tokens.

[![PyPI version](https://img.shields.io/pypi/v/tenro.svg)](https://pypi.org/project/tenro/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/tenro/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

[Documentation](https://tenro.ai/docs) ·
[Quickstart](https://tenro.ai/docs/getting-started/quickstart)

</div>

## ✨ Features

- **Simulate scenarios** — Control LLM responses, errors, and tool results
- **Verify workflows** — Check tool usage, edge cases, and agent behavior
- **Run evaluations** — Measure agent behavior across test cases
- **Agnostic by design** — Works across multiple LLM providers and agent frameworks
- **Multi-agent support** — Test multi-agent workflows

## Install

```bash
pip install tenro
```

## Quick Start

### 1. Decorate your tools and agent

Decorate with `@link_tool` and `@link_agent`.

```python
# myapp/agent.py
from tenro import link_agent, link_tool


def add_footer(text: str) -> str:
    # Post-processing before returning the final reply
    return f"{text}\n\nReply in the support portal for faster help."


@link_tool
def lookup_ticket(ticket_id: str) -> dict:
    return ticket_api.get(ticket_id)


@link_agent
class SupportAgent:
    def run(self, task: str) -> str:
        ...  # LLM calls tools and drafts the reply
        return add_footer(llm_response)
```

### 2. Simulate your test scenarios

Use the `@tenro.simulate` pytest decorator to simulate LLM calls and tool results.

```python
# tests/test_agent.py
import tenro
from tenro import Provider, ToolCall
from tenro.simulate import llm, tool
from myapp.agent import lookup_ticket, SupportAgent


@tenro.simulate
def test_agent():
    # Simulate the tool result
    tool.simulate(
        lookup_ticket,
        result={"status": "shipped", "eta": "Apr 18"}
    )

    # Simulate the LLM calls
    llm.simulate(
        Provider.OPENAI,
        responses=[
            # 1) call the tool
            ToolCall(lookup_ticket, ticket_id="T-123"),

            # 2) generate the reply
            "Your replacement has shipped and should arrive by Apr 18."
        ]
    )

    # Run your agent
    SupportAgent().run("What's the status of ticket T-123?")

    # Verify the tool was called
    tool.verify(lookup_ticket, ticket_id="T-123")

    # Verify what the user sees and the footer got added
    agent.verify(
        SupportAgent,
        result="Your replacement has shipped and should arrive by Apr 18.\n\n"
               "Reply in the support portal for faster help."
    )
```

## Trace Output

Tenro can print a local trace of your agent run so you can see each step of execution, including user input, LLM calls, tool calls, and the final output.

```text
🤖 SupportAgent
   ├─ → user: "What's the status of ticket T-123?"
   │
   ├─ 🧠 GPT-5.4 [SIM]
   │     ├─ → prompt: "What's the status of ticket T-123?"
   │     └─ ← tool_call: lookup_ticket(ticket_id='T-123')
   │
   ├─ 🔧 lookup_ticket [SIM]
   │     ├─ → ticket_id='T-123'
   │     └─ ← {'status': 'shipped', 'eta': 'Apr 18'}
   │
   ├─ 🧠 GPT-5.4
   │     ├─ → prompt: "Tool result: {'status': 'shipped', 'eta': 'Apr 18'}"
   │     └─ ← "Your replacement has shipped and should arrive by Apr 18."
   │
   └─ ← "Your replacement has shipped and should arrive by Apr 18.

         Reply in the support portal for faster help."

────────────────────────────────────────────────────────────────
Summary: 1 agent | 2 LLM calls | 1 tool call | Total: 355ms
```

Enable trace output locally:
> `TENRO_PRINT_TRACE=1`

## How It Works

Decorate your tools, agents, and model calls with `@link_tool`, `@link_agent`, and `@link_llm`, define the behavior you
want with Tenro’s simulation API, then run your tests normally.

Tenro helps you simulate and validate AI agents in two ways:

### 1. Test the agent code

Simulate LLM responses and verify that your code handles failures, retries, guardrails, and edge cases correctly.

- Invalid tool inputs
- Repeated or unexpected tool calls
- Retry, fallback, and escalation paths

### 2. Test the LLM behavior

Simulate tools and environment conditions to see how the model behaves under realistic scenarios.

- Tool choice and sequencing
- Incomplete or invalid tool results
- Ambiguous or inconsistent environments

## LLM Provider Support

| Provider  | API                   | Text |    Status    |
|-----------|-----------------------|:----:|:------------:|
| OpenAI    | Chat Completions API  | Yes  |  Supported   |
| Anthropic | Messages API          | Yes  |  Supported   |
| Gemini    | Generate Content API  | Yes  |  Supported   |
| Others    | OpenAI-compatible API | Yes  | Experimental |

## Agent Framework Support

| Framework     |  Status   |
|---------------|:---------:|
| LangChain     | Supported |
| Pydantic AI   | Supported |
| AutoGen       | Supported |
| LangGraph     | Supported |
| LlamaIndex    | Supported |
| CrewAI        | Supported |
| Custom Agents | Supported |

## Support

If Tenro helps you, consider starring the repo to bookmark it and help others discover Tenro.

[![Star the repo](https://img.shields.io/badge/⭐%20Star%20the%20repo-GitHub-black?style=for-the-badge)](https://github.com/tenro-ai/tenro-python)

- **Report bugs** — include exact steps and logs if possible
- **Request features** — share the use case and expected behavior
- **Ask questions** — usage, roadmap, or design decisions

Please use [GitHub Issues](https://github.com/tenro-ai/tenro-python/issues) for bug reports, feature requests, and
questions.

## Contributing

Thanks for your interest in contributing.

Tenro is evolving quickly, and the current focus is on stabilizing the core API. As a result, pull requests are not
being accepted at this time.

## License

[Apache 2.0](LICENSE)

## Contact

- Email: support@tenro.ai
