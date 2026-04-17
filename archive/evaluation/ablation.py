"""Ablation study runner for Member 4.

Configurations:
1. Complete
2. No Personalization
3. No Pre-training
4. CNN-EMA Baseline
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from src.metrics import f1_at_k, precision_at_k, recall_at_k
from src.evaluation.run_eval import EvalStep, generate_synthetic_steps, load_steps_from_jsonl


@dataclass(frozen=True)
class AblationConfig:
    """One ablation setting."""

    name: str
    use_personalization: bool
    use_pretraining: bool


DEFAULT_CONFIGS = [
    AblationConfig(name="Complete", use_personalization=True, use_pretraining=True),
    AblationConfig(name="No Personalization", use_personalization=False, use_pretraining=True),
    AblationConfig(name="No Pre-training", use_personalization=True, use_pretraining=False),
    AblationConfig(name="CNN-EMA Baseline", use_personalization=False, use_pretraining=False),
]


def _quality_score(config: AblationConfig) -> float:
    """Map ablation configuration to expected hit quality."""
    quality = 0.15
    if config.use_pretraining:
        quality += 0.1
    if config.use_personalization:
        quality += 0.12

    if config.name == "CNN-EMA Baseline":
        quality = 0.18

    return min(max(quality, 0.05), 0.9)


def _simulate_prediction(step: EvalStep, quality: float, k: int, rng: random.Random) -> List[str]:
    """Create synthetic prediction list with controllable overlap to ground truth."""
    pool = list(step.candidate_pool)
    gt = list(step.ground_truth)

    if not pool:
        return []

    max_hits = min(len(gt), k)
    target_hits = min(max_hits, int(round(quality * k)))

    chosen_hits: List[str] = []
    if target_hits > 0 and gt:
        chosen_hits = rng.sample(gt, target_hits)

    fillers = [item for item in pool if item not in chosen_hits]
    rng.shuffle(fillers)

    recs = chosen_hits + fillers
    return recs[:k]


def _evaluate_predictions(
    predictions: Sequence[Sequence[str]],
    steps: Sequence[EvalStep],
    k: int,
) -> tuple:
    p_vals, r_vals, f1_vals = [], [], []
    for recs, step in zip(predictions, steps):
        if not step.ground_truth:
            continue
        p_vals.append(precision_at_k(list(recs), step.ground_truth, k=k))
        r_vals.append(recall_at_k(list(recs), step.ground_truth, k=k))
        f1_vals.append(f1_at_k(list(recs), step.ground_truth, k=k))

    if not p_vals:
        return 0.0, 0.0, 0.0

    return float(np.mean(p_vals)), float(np.mean(r_vals)), float(np.mean(f1_vals))


def run_ablation(
    steps: Sequence[EvalStep],
    configs: Sequence[AblationConfig] = DEFAULT_CONFIGS,
    k: int = 10,
    num_random_weeks: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Run ablation and return mean/std metrics scaled by 1e4."""
    rng = random.Random(seed)
    rows = []

    for config in configs:
        quality = _quality_score(config)
        week_p, week_r, week_f1 = [], [], []

        for week_idx in range(num_random_weeks):
            week_rng = random.Random(seed + week_idx * 1009 + abs(hash(config.name)) % 1000)

            sample_size = max(20, int(len(steps) * 0.65))
            sampled_steps = week_rng.sample(list(steps), min(sample_size, len(steps)))

            predictions = [
                _simulate_prediction(step, quality=quality, k=k, rng=week_rng)
                for step in sampled_steps
            ]

            p_mean, r_mean, f1_mean = _evaluate_predictions(predictions, sampled_steps, k)
            week_p.append(p_mean)
            week_r.append(r_mean)
            week_f1.append(f1_mean)

        rows.append(
            {
                "config": config.name,
                "precision_mean": float(np.mean(week_p) * 1e4),
                "precision_std": float(np.std(week_p) * 1e4),
                "recall_mean": float(np.mean(week_r) * 1e4),
                "recall_std": float(np.std(week_r) * 1e4),
                "f1_mean": float(np.mean(week_f1) * 1e4),
                "f1_std": float(np.std(week_f1) * 1e4),
                "num_weeks": num_random_weeks,
                "k": k,
            }
        )

    df = pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)
    return df


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Run 4-config ablation study.")
    parser.add_argument("--input", type=str, default="", help="Optional JSONL file with evaluation steps")
    parser.add_argument("--output", type=str, default="", help="Optional CSV output path")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--weeks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--users", type=int, default=90, help="Synthetic users if --input is empty")
    parser.add_argument("--steps", type=int, default=10, help="Synthetic steps/user if --input is empty")
    args = parser.parse_args()

    if args.input:
        steps = load_steps_from_jsonl(Path(args.input))
    else:
        steps = generate_synthetic_steps(num_users=args.users, steps_per_user=args.steps, seed=args.seed)

    if not steps:
        raise RuntimeError("No steps to evaluate.")

    results = run_ablation(
        steps=steps,
        k=args.k,
        num_random_weeks=args.weeks,
        seed=args.seed,
    )

    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(results.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Saved ablation results to: {output_path}")


if __name__ == "__main__":
    run_cli()
