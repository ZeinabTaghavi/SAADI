"""Output path conventions for SAADI generation experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SaadiGenerationPaths:
    dataset_experiment_dir: Path
    generation_pipeline_dir: Path
    saadi_generation_dir: Path
    experiment_output_dir: Path
    generations_jsonl: Path
    context_debug_jsonl: Path
    generated_bridges_jsonl: Path
    run_config_json: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def resolve_saadi_generation_paths(
    *,
    dataset_experiment_dir: str | Path,
    experiment_name: str,
    dataset: str,
    retriever: str,
    top_k: int,
    summary_k: int | None = None,
    output_name: str | None = None,
) -> SaadiGenerationPaths:
    root = Path(dataset_experiment_dir).resolve()
    generation_pipeline_dir = root / "generation_pipeline"
    saadi_generation_dir = generation_pipeline_dir / "saadi_generation"
    if output_name:
        experiment_output_dir = saadi_generation_dir / "outputs" / str(output_name)
    else:
        experiment_output_dir = (
            saadi_generation_dir
            / "outputs"
            / str(experiment_name)
            / f"top_{int(top_k)}"
            / f"dataset={dataset}"
            / f"retriever={retriever}"
        )
    if not output_name and summary_k is not None and int(summary_k) > 0:
        experiment_output_dir = experiment_output_dir / f"summary_k={int(summary_k)}"
    return SaadiGenerationPaths(
        dataset_experiment_dir=root,
        generation_pipeline_dir=generation_pipeline_dir,
        saadi_generation_dir=saadi_generation_dir,
        experiment_output_dir=experiment_output_dir,
        generations_jsonl=experiment_output_dir / "generations.jsonl",
        context_debug_jsonl=experiment_output_dir / "context_debug.jsonl",
        generated_bridges_jsonl=experiment_output_dir / "generated_bridges.jsonl",
        run_config_json=experiment_output_dir / "run_config.json",
    )
