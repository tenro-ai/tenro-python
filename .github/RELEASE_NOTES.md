🎉 **OTel-aligned span model**

✨ **Added**

- **OTel-aligned spans**: Spans now include `trace_id`, `parent_id`, `status`, and nanosecond-precision timestamps following OpenTelemetry semantic conventions
- **`SpanHandler` protocol**: Pluggable interface for custom span backends

⚡ **Changed**

- **Nanosecond timestamps**: Span timestamps now use `int` nanoseconds (previously `float` seconds)
- **Lazy-loaded imports**: `import tenro` no longer eagerly imports all submodules, reducing startup time

---

**Installation**

```bash
pip install tenro==0.2.2
```
