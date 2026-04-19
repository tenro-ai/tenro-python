🎉 **SSE streaming, trajectory assertions, and simulation improvements**

✨ **Added**

- **`@tenro.simulate` decorator**: Simplified test setup — annotate your test function directly without explicit fixture parameters
- **SSE streaming simulation**: Simulate streaming responses for OpenAI, Anthropic, and Gemini providers
- **Verify agent trajectories**: `assert_trajectory_contains()`, `assert_called_before()`, and `assert_called()` let you assert the exact sequence of LLM calls your agent made
- **Content capture**: Opt-in capture of LLM request/response content on spans
- **`allow_real_llm_calls`**: Permit live LLM calls in specific tests without disabling simulation globally
- **Agent identity fields**: `link_agent()` now accepts `agent_id`, `agent_name`, and `agent_version`
- **`[SIM]` marker in trace output**: Simulated spans are visually distinguished in trace output. Set `TENRO_PRINT_TRACE_SIM_MARKER=false` to hide it

🔄 **Renamed**

- **`strict_expectations` → `fail_unused`**: The `Construct` parameter, decorator option, pytest flag, and env var have been renamed for clarity:
  - `Construct(strict_expectations=True)` → `Construct(fail_unused=True)`
  - `@tenro.simulate(strict_expectations=True)` → `@tenro.simulate(fail_unused=True)`
  - `--tenro-strict-expectations` → `--tenro-fail-unused`
  - `TENRO_STRICT_EXPECTATIONS=1` → `TENRO_FAIL_UNUSED=1`

⚠️ **Deprecated**

- **`@tenro` decorator**: Use `@tenro.simulate` instead — `@tenro` will be removed in a future release

💪 **Fixed**

- Test isolation improved: simulation state is correctly reset between test runs

---

**Installation**

```bash
pip install tenro==0.2.3
```
