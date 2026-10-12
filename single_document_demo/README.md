# One-document Saadi example

Run from the repository root after installing the package:

```bash
python single_document_demo/run.py text --mock --query 'Who leads the Aurora team?'
python single_document_demo/run.py therapy --mock
```

The first example reads `input/sample_text/document.txt`. The second uses the `therapy` conversation profile with the synthetic `input/sample_conversation/conversation.csv`. No participant or patient records are included. The mock backend checks the workflow on CPU; it does not provide meaningful model answers or an experimental performance estimate.

To process your document with a real language model, pick a provider with `--llm`:

```bash
python single_document_demo/run.py text --llm openai:gpt-4.1-mini \
  --input /absolute/path/document.txt --query 'What is the document about?'
python single_document_demo/run.py text --llm ollama:qwen3:8b --query 'Who leads the Aurora team?'
python single_document_demo/run.py text --llm anthropic:claude-opus-5-5 --query 'Who leads the Aurora team?'
```

Hosted providers read their key from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`); Ollama needs none. Without `--llm`, the `[model]` table of the TOML configuration applies, which defaults to local vLLM. The [backend table](../README.md#language-model-backends) in the main README lists the install extras.

TOML configurations are in `configs/`. Use `--config` for your own settings, `--output-root` to choose a destination, or `--questions` for a question JSONL file. Conversation CSVs use `turn_id`, `role`, and `text`; roles are `therapist` and `patient` for this profile.

Outputs are written under `single_document_demo/output/`, including construction artifacts, readable cleaned outputs, and batch manifests. This directory is excluded from Git. The example imports the same Saadi implementation as the study pipeline.
