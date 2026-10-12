from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from saadi.pipelines.single_document_pipeline import SingleDocumentPipelineConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_ROOT = REPO_ROOT / "single_document_demo"
WORKFLOW_TEXT = "text"
WORKFLOW_THERAPY = "therapy"
SUPPORTED_WORKFLOWS = (WORKFLOW_TEXT, WORKFLOW_THERAPY)


def default_model_config() -> dict[str, Any]:
    return {
        "backend": "vllm",
        "model_id": "Qwen/Qwen3.5-27B",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "load_chat_tokenizer": True,
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 32768,
        "enforce_eager": False,
        "vllm_kwargs": {
            "disable_custom_all_reduce": True,
            "max_num_batched_tokens": 16384,
            "max_num_seqs": 16,
        },
        "sampling": {"top_p": 0.95},
    }


@dataclass(slots=True)
class DemoConfig:
    config_path: Path
    workflow: str
    demo_root: Path = DEFAULT_DEMO_ROOT
    input_path: Path = DEFAULT_DEMO_ROOT / "input" / "sample_text" / "document.txt"
    output_root: Path = DEFAULT_DEMO_ROOT / "output"
    prompt_profile: str = "default"
    role_column: str = "role"
    text_column: str = "text"
    order_column: str = "turn_id"
    allowed_roles: tuple[str, ...] = ("therapist", "patient")
    role_aliases: dict[str, str] = field(default_factory=dict)
    include_timestamps: bool = False
    run_name: str = "text_document"
    query: str = ""
    questions_path: str = ""
    model_config: dict[str, Any] = field(default_factory=default_model_config)
    saadi_options: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_root(self) -> Path:
        return self.demo_root / "memory_prompts"

    @property
    def input_format(self) -> str:
        return "conversation_csv" if self.workflow == WORKFLOW_THERAPY else "text"


@dataclass(slots=True)
class PromptedPipelineConfig(SingleDocumentPipelineConfig):
    output_dir: str = ""
    memory_prompt_profile: str = "default"
    memory_prompt_hash: str = ""
    memory_entity_system_prompt: str = ""
    memory_rollup_system_prompt: str = ""
    document_summary_enabled: bool = False
    document_summary_max_tokens: int = 384
    document_summary_temperature: float = 0.2

    def run_dir(self) -> Path:
        """Keep demo output routing local without changing core SAADI."""

        if self.output_dir:
            return Path(self.output_dir)
        return SingleDocumentPipelineConfig.run_dir(self)

    def strategy_kwargs(self) -> dict[str, Any]:
        kwargs = SingleDocumentPipelineConfig.strategy_kwargs(self)
        kwargs["system_prompt"] = self.memory_entity_system_prompt
        kwargs["rollup_system_prompt"] = self.memory_rollup_system_prompt
        return kwargs


def _repo_relative_path(value: str | Path, *, default: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_demo_config(path: str | Path) -> DemoConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Demo config does not exist: {config_path}")
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)

    demo = payload.get("demo", {}) or {}
    workflow_cfg = payload.get("workflow", {}) or {}
    input_cfg = payload.get("input", {}) or {}
    output_cfg = payload.get("output", {}) or {}
    run = payload.get("run", {}) or {}
    model = payload.get("model", {}) or {}
    saadi = payload.get("saadi", {}) or {}

    workflow = str(workflow_cfg.get("type", "") or "").strip().lower()
    if workflow not in SUPPORTED_WORKFLOWS:
        raise ValueError(
            f"Invalid [workflow].type {workflow!r}; expected one of {list(SUPPORTED_WORKFLOWS)}"
        )

    demo_root = _repo_relative_path(
        demo.get("root", "single_document_demo"), default=DEFAULT_DEMO_ROOT
    )
    default_input = (
        demo_root / "input"
        if workflow == WORKFLOW_THERAPY
        else demo_root / "input" / "sample_text" / "document.txt"
    )
    default_profile = "therapy" if workflow == WORKFLOW_THERAPY else "default"
    default_run_name = "therapy_conversations" if workflow == WORKFLOW_THERAPY else "text_document"

    model_config = default_model_config()
    model_config.update(model)
    if str(model_config.get("backend", "")).lower() == "mock":
        model_config = {"backend": "mock"}

    allowed_roles = tuple(
        str(role).strip().casefold()
        for role in input_cfg.get("allowed_roles", ["therapist", "patient"])
        if str(role).strip()
    )
    role_aliases = {
        str(key).strip().casefold(): str(value).strip().casefold()
        for key, value in (input_cfg.get("role_aliases", {}) or {}).items()
    }
    return DemoConfig(
        config_path=config_path.resolve(),
        workflow=workflow,
        demo_root=demo_root.resolve(),
        input_path=_repo_relative_path(input_cfg.get("path", ""), default=default_input).resolve(),
        output_root=_repo_relative_path(
            output_cfg.get("root", ""), default=demo_root / "output"
        ).resolve(),
        prompt_profile=str(
            workflow_cfg.get("prompt_profile", default_profile) or default_profile
        ).strip(),
        role_column=str(input_cfg.get("role_column", "role") or "role"),
        text_column=str(input_cfg.get("text_column", "text") or "text"),
        order_column=str(input_cfg.get("order_column", "turn_id") or "turn_id"),
        allowed_roles=allowed_roles,
        role_aliases=role_aliases,
        include_timestamps=bool(input_cfg.get("include_timestamps", False)),
        run_name=str(run.get("name", default_run_name) or default_run_name).strip(),
        query=str(run.get("query", "") or "").strip(),
        questions_path=str(run.get("questions_path", "") or "").strip(),
        model_config=model_config,
        saadi_options=dict(saadi),
    )
