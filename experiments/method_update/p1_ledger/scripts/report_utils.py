"""Metric and table helpers shared by steps 3-5 (bootstrap, McNemar, side-by-side formatting)."""

from __future__ import annotations

import statistics
from typing import Any, Iterable

from common import N_RESAMPLES, SEED, mcnemar_exact_p, paired_bootstrap_ci

METHODS = ("vanilla", "no_memory", "frozen", "live")
METHOD_LABELS = {"vanilla": "Vanilla", "no_memory": "No memory", "frozen": "Frozen", "live": "Live"}


def hr(group: list[dict], method: str, k: str = "10") -> float:
    return 100.0 * sum(bool(p.get(f"hit{k}_{method}")) for p in group) / len(group) if group else float("nan")


def median_rank(group: list[dict], key: str) -> float:
    vals = [p[key] for p in group if p.get(key) is not None]
    return float(statistics.median(vals)) if vals else float("nan")


def rate(group: list[dict], key: str) -> tuple[int, int]:
    return sum(1 for p in group if p.get(key)), len(group)


def rate_text(group: list[dict], key: str) -> str:
    n, d = rate(group, key)
    return f"{n}/{d} ({100.0 * n / d:.0f}%)" if d else "-"


def side_by_side(run1: str, run0: str | None) -> str:
    return f"{run1} (run 0: {run0})" if run0 is not None else f"{run1} (run 0: n/a)"


def bootstrap_mean(values: Iterable[float], *, seed: int = SEED, n_resamples: int = N_RESAMPLES) -> tuple[float, float, float]:
    return paired_bootstrap_ci(list(values), n_resamples=n_resamples, seed=seed)


def hr_with_ci(group: list[dict], method: str, k: str) -> str:
    mean, lo, hi = bootstrap_mean([100.0 * float(bool(p.get(f"hit{k}_{method}"))) for p in group])
    return f"{mean:.0f} [{lo:.0f}, {hi:.0f}]"


def paired_text(group: list[dict], a: str, b: str, k: str = "10") -> str:
    diffs = [(float(bool(p.get(f"hit{k}_{a}"))) - float(bool(p.get(f"hit{k}_{b}")))) * 100.0 for p in group]
    mean, lo, hi = bootstrap_mean(diffs)
    x = sum(1 for p in group if p.get(f"hit{k}_{a}") and not p.get(f"hit{k}_{b}"))
    y = sum(1 for p in group if p.get(f"hit{k}_{b}") and not p.get(f"hit{k}_{a}"))
    return f"{mean:+.1f} pp [{lo:+.1f}, {hi:+.1f}], McNemar p={mcnemar_exact_p(x, y):.3f} ({x} vs {y})"


def paired_numbers(group: list[dict], a: str, b: str, k: str = "10") -> dict[str, Any]:
    diffs = [(float(bool(p.get(f"hit{k}_{a}"))) - float(bool(p.get(f"hit{k}_{b}")))) * 100.0 for p in group]
    mean, lo, hi = bootstrap_mean(diffs)
    x = sum(1 for p in group if p.get(f"hit{k}_{a}") and not p.get(f"hit{k}_{b}"))
    y = sum(1 for p in group if p.get(f"hit{k}_{b}") and not p.get(f"hit{k}_{a}"))
    return {"mean_pp": mean, "ci_lo": lo, "ci_hi": hi, "discordant_a_only": x, "discordant_b_only": y, "mcnemar_p": mcnemar_exact_p(x, y)}


def seeded_sample(rows: list[Any], n: int, *, seed: int = SEED) -> list[Any]:
    import numpy as np

    if not rows:
        return []
    rng = np.random.default_rng(seed)
    picks = sorted(int(i) for i in rng.choice(len(rows), size=min(n, len(rows)), replace=False))
    return [rows[i] for i in picks]
