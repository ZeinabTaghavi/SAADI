from __future__ import annotations

from pathlib import Path

from single_document_demo.pipeline.config import default_model_config, resolve_model_config
from single_document_demo.run import build_arg_parser, main


def test_resolve_model_config_applies_vllm_defaults_only_to_vllm() -> None:
    assert resolve_model_config({"backend": "mock", "tensor_parallel_size": 4}) == {"backend": "mock"}

    vllm = resolve_model_config({"model_id": "org/model"})
    assert vllm["backend"] == "vllm"
    assert vllm["model_id"] == "org/model"
    assert vllm["tensor_parallel_size"] == default_model_config()["tensor_parallel_size"]

    api = resolve_model_config(
        {"backend": "openai-compatible", "model_id": "gpt-4.1-mini", "base_url": "http://localhost:8000/v1"}
    )
    assert api == {"backend": "openai", "model_id": "gpt-4.1-mini", "base_url": "http://localhost:8000/v1"}


def test_cli_llm_option_parses_provider_and_model() -> None:
    args = build_arg_parser().parse_args(["text", "--llm", "ollama:qwen3:8b"])
    assert args.llm == "ollama:qwen3:8b"
    assert args.mock is False


def test_cli_runs_the_text_workflow_with_the_mock_llm(tmp_path: Path) -> None:
    exit_code = main(
        [
            "text",
            "--llm",
            "mock",
            "--query",
            "Who leads the Aurora team?",
            "--output-root",
            str(tmp_path),
            "--run-name",
            "cli_mock",
        ]
    )
    assert exit_code == 0
    generation = tmp_path / "sample_text" / "generation"
    assert (generation / "answer.txt").is_file()
    prompt_text = (generation / "prompt.txt").read_text(encoding="utf-8")
    assert "<|system|>" in prompt_text
    assert "Who leads the Aurora team?" in prompt_text
