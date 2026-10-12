# One-document Saadi example

Run from the repository root after installing the package:

```bash
python single_document_demo/run.py text --mock --query 'Who leads the Aurora team?'
python single_document_demo/run.py therapy --mock
```

The first example reads `input/sample_text/document.txt`. The second uses the `therapy` conversation profile with the synthetic `input/sample_conversation/conversation.csv`. No participant or patient records are included. The mock backend checks the workflow on CPU; it does not provide meaningful model answers or an experimental performance estimate.

To process your document with the configured language model:

```bash
python single_document_demo/run.py text --input /absolute/path/document.txt \
  --query 'What is the document about?'
```

TOML configurations are in `configs/`. Use `--config` for your own settings, `--output-root` to choose a destination, or `--questions` for a question JSONL file. Conversation CSVs use `turn_id`, `role`, and `text`; roles are `therapist` and `patient` for this profile.

Outputs are written under `single_document_demo/output/`, including construction artifacts, readable cleaned outputs, and batch manifests. This directory is excluded from Git. The example imports the same Saadi implementation as the study pipeline.
