"""Evaluation runner for recommendation quality.

This module focuses on Member 4 responsibilities:
- P@10, R@10, F1@10 evaluation utilities
- Baselines: Random, Last-K, CNN-EMA
- Lightweight CLI for synthetic or JSONL inputs
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import f1_at_k, precision_at_k, recall_at_k


@dataclass
class EvalStep:
    """Single evaluation step for one user interaction point."""

    user_id: str
    ground_truth: List[str]
    candidate_pool: List[str]
    history: List[str]
    ema_vector: Optional[List[float]] = None
    model_recommended: Optional[List[str]] = None


def _vector_from_item(item_id: str, dim: int = 32) -> np.ndarray:
    """Create deterministic item vectors so baselines can run without artifacts."""
    rng = np.random.default_rng(abs(hash(item_id)) % (2**32))
    vec = rng.normal(0.0, 1.0, size=dim)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def random_baseline(candidate_pool: Sequence[str], k: int, rng: random.Random) -> List[str]:
    """Random baseline from current candidate pool."""
    if not candidate_pool:
        return []
    sample_size = min(k, len(candidate_pool))
    return rng.sample(list(candidate_pool), sample_size)


def last_k_baseline(history: Sequence[str], candidate_pool: Sequence[str], k: int) -> List[str]:
    """Recommend the most recent interacted items (if still in candidate pool)."""
    recs: List[str] = []
    pool_set = set(candidate_pool)

    for item in reversed(history):
        if item in pool_set and item not in recs:
            recs.append(item)
            if len(recs) >= k:
                return recs

    for item in candidate_pool:
        if item not in recs:
            recs.append(item)
            if len(recs) >= k:
                break

    return recs


def cnn_ema_baseline(
    ema_vector: Optional[Sequence[float]],
    candidate_pool: Sequence[str],
    k: int,
) -> List[str]:
    """Simple nearest-neighbor ranking between EMA and candidate item vectors."""
    if not candidate_pool:
        return []

    if ema_vector is None:
        return list(candidate_pool[:k])

    ema = np.asarray(ema_vector, dtype=np.float32)
    dim = len(ema)

    scored = []
    for item_id in candidate_pool:
        item_vec = _vector_from_item(item_id, dim=dim)
        dist = float(np.linalg.norm(ema - item_vec))
        scored.append((dist, item_id))

    scored.sort(key=lambda row: row[0])
    return [item_id for _, item_id in scored[:k]]


def evaluate_prediction_set(
    recommendations: Sequence[Sequence[str]],
    ground_truths: Sequence[Sequence[str]],
    k: int = 10,
    scale_factor: int = 10000,
) -> Dict[str, float]:
    """Aggregate P@K/R@K/F1@K for a prediction set."""
    p_values: List[float] = []
    r_values: List[float] = []
    f1_values: List[float] = []

    for recs, gt in zip(recommendations, ground_truths):
        if not gt:
            continue
        p_values.append(precision_at_k(list(recs), list(gt), k=k))
        r_values.append(recall_at_k(list(recs), list(gt), k=k))
        f1_values.append(f1_at_k(list(recs), list(gt), k=k))

    if not p_values:
        return {
            "precision_mean": 0.0,
            "precision_std": 0.0,
            "recall_mean": 0.0,
            "recall_std": 0.0,
            "f1_mean": 0.0,
            "f1_std": 0.0,
            "n_steps": 0,
        }

    return {
        "precision_mean": float(np.mean(p_values) * scale_factor),
        "precision_std": float(np.std(p_values) * scale_factor),
        "recall_mean": float(np.mean(r_values) * scale_factor),
        "recall_std": float(np.std(r_values) * scale_factor),
        "f1_mean": float(np.mean(f1_values) * scale_factor),
        "f1_std": float(np.std(f1_values) * scale_factor),
        "n_steps": len(p_values),
    }


def evaluate_steps_with_strategy(
    steps: Sequence[EvalStep],
    strategy: Callable[[EvalStep], List[str]],
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate one strategy over all evaluation steps."""
    recs = [strategy(step) for step in steps]
    gts = [step.ground_truth for step in steps]
    return evaluate_prediction_set(recommendations=recs, ground_truths=gts, k=k)


def evaluate_baselines(
    steps: Sequence[EvalStep],
    k: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Run three baseline methods and return comparable metrics."""
    rng = random.Random(seed)

    random_result = evaluate_steps_with_strategy(
        steps,
        strategy=lambda s: random_baseline(s.candidate_pool, k=k, rng=rng),
        k=k,
    )

    lastk_result = evaluate_steps_with_strategy(
        steps,
        strategy=lambda s: last_k_baseline(s.history, s.candidate_pool, k=k),
        k=k,
    )

    cnn_ema_result = evaluate_steps_with_strategy(
        steps,
        strategy=lambda s: cnn_ema_baseline(s.ema_vector, s.candidate_pool, k=k),
        k=k,
    )

    rows = [
        {"model": "Random", **random_result},
        {"model": "Last-K", **lastk_result},
        {"model": "CNN-EMA", **cnn_ema_result},
    ]

    return pd.DataFrame(rows)


def evaluate_optional_model_predictions(steps: Sequence[EvalStep], k: int = 10) -> Optional[Dict[str, float]]:
    """Evaluate model predictions when present in input JSONL."""
    available = [step for step in steps if step.model_recommended]
    if not available:
        return None

    recommendations = [step.model_recommended or [] for step in available]
    ground_truths = [step.ground_truth for step in available]
    return evaluate_prediction_set(recommendations, ground_truths, k=k)


def generate_synthetic_steps(
    num_users: int = 80,
    steps_per_user: int = 10,
    catalog_size: int = 1000,
    candidate_pool_size: int = 120,
    ground_truth_size: int = 12,
    seed: int = 42,
) -> List[EvalStep]:
    """Create synthetic sessions so evaluation code works before real logs are available."""
    rng = random.Random(seed)
    catalog = [f"{100000000 + i}" for i in range(catalog_size)]

    steps: List[EvalStep] = []

    for user_idx in range(num_users):
        user_id = f"user_{user_idx:04d}"
        user_pref = set(rng.sample(catalog, 80))
        history: List[str] = []

        for _ in range(steps_per_user):
            candidate_pool = rng.sample(catalog, min(candidate_pool_size, len(catalog)))
            preferred_candidates = [item for item in candidate_pool if item in user_pref]

            if preferred_candidates:
                gt_count = min(ground_truth_size, len(preferred_candidates))
                ground_truth = rng.sample(preferred_candidates, gt_count)
            else:
                ground_truth = rng.sample(candidate_pool, min(ground_truth_size, len(candidate_pool)))

            if history:
                recent = history[-5:]
                vecs = [_vector_from_item(item_id) for item_id in recent]
                ema_vector = np.mean(vecs, axis=0).tolist()
            else:
                ema_vector = None

            steps.append(
                EvalStep(
                    user_id=user_id,
                    ground_truth=ground_truth,
                    candidate_pool=candidate_pool,
                    history=list(history),
                    ema_vector=ema_vector,
                )
            )

            if ground_truth:
                history.append(ground_truth[0])

    return steps


def load_steps_from_jsonl(path: Path) -> List[EvalStep]:
    """Load evaluation steps from JSONL file.

    Supported keys per line:
    - user_id
    - ground_truth
    - candidate_pool
    - history (optional)
    - ema_vector (optional)
    - model_recommended or recommended (optional)
    """
    steps: List[EvalStep] = []

    with path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            recommended = payload.get("model_recommended") or payload.get("recommended")
            steps.append(
                EvalStep(
                    user_id=str(payload["user_id"]),
                    ground_truth=[str(x) for x in payload.get("ground_truth", [])],
                    candidate_pool=[str(x) for x in payload.get("candidate_pool", [])],
                    history=[str(x) for x in payload.get("history", [])],
                    ema_vector=payload.get("ema_vector"),
                    model_recommended=[str(x) for x in recommended] if recommended else None,
                )
            )

    return steps


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Run recommendation evaluation baselines.")
    parser.add_argument("--input", type=str, default="", help="Optional JSONL input path")
    parser.add_argument("--output", type=str, default="", help="Optional CSV output path")
    parser.add_argument("--k", type=int, default=10, help="Top-k used for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--users", type=int, default=80, help="Synthetic users when --input is empty")
    parser.add_argument("--steps", type=int, default=10, help="Synthetic steps per user when --input is empty")
    args = parser.parse_args()

    if args.input:
        steps = load_steps_from_jsonl(Path(args.input))
    else:
        steps = generate_synthetic_steps(num_users=args.users, steps_per_user=args.steps, seed=args.seed)

    if not steps:
        raise RuntimeError("No evaluation steps were loaded.")

    baseline_df = evaluate_baselines(steps=steps, k=args.k, seed=args.seed)

    model_metrics = evaluate_optional_model_predictions(steps=steps, k=args.k)
    if model_metrics is not None:
        baseline_df = pd.concat(
            [
                baseline_df,
                pd.DataFrame([{"model": "ModelInput", **model_metrics}]),
            ],
            ignore_index=True,
        )

    baseline_df = baseline_df.sort_values(by="f1_mean", ascending=False).reset_index(drop=True)

    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(baseline_df.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_df.to_csv(output_path, index=False)
        print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    run_cli()
