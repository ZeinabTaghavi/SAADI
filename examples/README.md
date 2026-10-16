# Examples

- `sample_text/document.txt`: a short synthetic paragraph; one chunk at the default size.
- `aurora_report/document.txt`: a 900-word synthetic report about a fictional research team. With `--chunk-size 100` it becomes about ten chunks and two fold summaries, so the entity memory and the hierarchy are both exercised. `saadi demo` uses a bundled copy of it.
- `sample_conversation/conversation.csv`: a tiny role-labelled transcript for the `conversation` prompt profile.
- `quickstart.ipynb`: the Python API end to end.

```bash
saadi index examples/aurora_report/document.txt --llm openai:gpt-4.1-mini --state state/aurora --chunk-size 100
saadi query "Who keeps the Corvin station?" --state state/aurora --llm openai:gpt-4.1-mini
saadi index examples/sample_conversation/conversation.csv --llm openai:gpt-4.1-mini --prompts conversation --state state/session --summary
```

All inputs are synthetic; no real people or records appear in them.
