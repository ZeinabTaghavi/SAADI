# Examples

Three notebooks, all openable in Colab from the repository page:

| Notebook | Runs on | What it shows |
|---|---|---|
| [quickstart.ipynb](quickstart.ipynb) | any runtime, an OpenAI (or compatible) key | index once, the entity memory and summary hierarchy, retrieve and answer, append without rebuilding, load the state again |
| [colab_local_model.ipynb](colab_local_model.ipynb) | Colab's free T4 GPU, no key | the same path with an open 8B model served by Ollama, with notes on what a small model does well and badly |
| [paper_results.ipynb](paper_results.ipynb) | any CPU runtime | rebuilds every table of the paper from the saved, hash-verified study values; nothing is recomputed |

Sample inputs:

- `aurora_report/document.txt`: a 900-word synthetic report about a fictional research team. With `--chunk-size 100` it becomes about ten chunks and two fold summaries, so the entity memory and the hierarchy are both exercised. `saadi demo` uses a bundled copy of it.
- `sample_text/document.txt`: a short synthetic paragraph; one chunk at the default size.
- `sample_conversation/conversation.csv`: a six-turn synthetic therapy dialogue, the same one `saadi check-prompts conversation` uses. With `--prompts conversation` the model builds a compact case-level memory from it rather than a list of names.

```bash
saadi index examples/aurora_report/document.txt --llm openai:gpt-4.1-mini --state saadi_state/aurora --chunk-size 100
saadi query "Who keeps the Corvin station?" --state saadi_state/aurora --llm openai:gpt-4.1-mini
saadi index examples/sample_conversation/conversation.csv --llm openai:gpt-4.1-mini --prompts conversation --state saadi_state/session --summary
```

All inputs are synthetic; no real people or records appear in them.
