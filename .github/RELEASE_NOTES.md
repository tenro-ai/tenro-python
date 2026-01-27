🎉 **New @tenro decorator and LLM tool call API**

✨ **Added**

- **`@tenro` decorator**: Simplified test setup - no explicit fixture parameter needed:
  ```python
  from tenro.testing import tenro
  from tenro.simulate import llm

  @tenro
  def test_my_agent():
      llm.simulate(response="Hello!")
      result = my_agent.run("prompt")
  ```

- **`LLMResponse`**: Ordered sequence of text and tool calls with full interleaving support:
  ```python
  from tenro import LLMResponse, ToolCall

  llm.simulate(Provider.ANTHROPIC, responses=[
      LLMResponse(blocks=["Let me search", ToolCall(search, query="AI")])
  ])
  ```

- **`RawLLMResponse`**: Raw provider JSON passthrough for edge cases:
  ```python
  from tenro import RawLLMResponse

  llm.simulate(Provider.OPENAI, responses=[
      RawLLMResponse(payload={"choices": [{"message": {"content": "Hi"}}]})
  ])
  ```

- **`ToolCall` smart constructor**: Direct tool call simulation without dict wrappers:
  ```python
  # Before: {"tool_calls": [ToolCall(search, query="AI")]}
  # After:
  llm.simulate(Provider.OPENAI, responses=[
      ToolCall(search, query="AI")
  ])
  ```

---

**Installation**

```bash
pip install tenro==0.2.1
```
