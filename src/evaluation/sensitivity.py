"""
Hyperparameter Sensitivity Analysis — Member 2 Task 8.

Sweeps alpha, sgd_steps, margin, adaptation_batch_size, personalization lifespan
using SessionSimulator on fake_behavior.csv + HGNN embeddings.

Usage:
    python -m src.evaluation.sensitivity
    python -m src.evaluation.sensitivity --output results/sensitivity.csv
"""

import argparse
import itertools
import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import torch

from src.inference.mlp_lifecycle import MLPLifecycleManager, LocalDictBackend
from src.inference.session_simulator import SessionSimulator
from src.inference.user_state import UserState
from src.models.personal_mlp import PersonalMLP, PersonalMLPFactory
from src.training.adapt_personal_mlp import TripletAdaptation
from src.evaluation.metrics import precision_at_k, recall_at_k, f1_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Artifact loading
# ============================================================

def load_hgnn_embeddings(pt_path: str) -> Dict[str, torch.Tensor]:
    """
    Load HGNN embeddings from article_embeddings_hgnn3.pt.

    Format: {'embeddings': Tensor[N, 64], 'id2idx': Dict[article_id, int]}
    article_id in file uses 10-digit zero-padded format.

    Returns:
        Dict[article_id_9digit, Tensor[64]]  — normalized to 9-digit to match behavior CSV.
    """
    data = torch.load(pt_path, weights_only=True)
    embeddings_matrix = data["embeddings"]   # [N, 64]
    id2idx: Dict[str, int] = data["id2idx"]

    result = {}
    for article_id_10, idx in id2idx.items():
        # Normalize: strip leading zero → 9-digit (matches fake_behavior.csv)
        article_id_9 = str(int(article_id_10))
        result[article_id_9] = embeddings_matrix[idx]

    logger.info("Loaded %d HGNN embeddings from %s", len(result), pt_path)
    return result


def load_behavior(csv_path: str, max_users: int = None) -> pd.DataFrame:
    """
    Load fake_behavior.csv.

    Columns: t_dat, customer_id, article_id, event_type
    Filters to purchase + cart events (positive signals).
    """
    df = pd.read_csv(csv_path)
    df["article_id"] = df["article_id"].astype(str)
    df["customer_id"] = df["customer_id"].astype(str)
    df["t_dat"] = pd.to_datetime(df["t_dat"])

    # Keep purchase and cart as positive interactions (click = weak signal)
    df = df[df["event_type"].isin(["purchase", "cart", "click"])].copy()
    df = df.sort_values("t_dat")

    if max_users is not None:
        top_users = df["customer_id"].value_counts().head(max_users).index
        df = df[df["customer_id"].isin(top_users)]

    logger.info(
        "Loaded behavior: %d rows, %d users, %d articles",
        len(df), df["customer_id"].nunique(), df["article_id"].nunique(),
    )
    return df


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_simulator_results(
    results: Dict[str, List[dict]],
    hgnn_embeddings: Dict[str, torch.Tensor],
    personal_mlp_template,
    k: int = 10,
) -> Dict[str, float]:
    """
    Compute mean P@K, R@K, F1@K from simulator output records.

    Each record has 'ground_truth' (list of future article_ids).
    We use the EMA vector at that point to get recommendations.
    """
    all_p, all_r, all_f1 = [], [], []

    for user_id, records in results.items():
        for rec in records:
            gt = rec.get("ground_truth", [])
            if not gt:
                continue
            # ground truth filtered to articles with embeddings
            gt_valid = [a for a in gt if a in hgnn_embeddings]
            if not gt_valid:
                continue

            # Recommendations stored if available
            recs = rec.get("recommendations", [])
            if recs:
                all_p.append(precision_at_k(recs, gt_valid, k))
                all_r.append(recall_at_k(recs, gt_valid, k))
                all_f1.append(f1_at_k(recs, gt_valid, k))

    n = len(all_p)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_evals": 0}

    return {
        "precision": sum(all_p) / n * 1e4,
        "recall": sum(all_r) / n * 1e4,
        "f1": sum(all_f1) / n * 1e4,
        "n_evals": n,
    }


def run_single_config(
    behavior_df: pd.DataFrame,
    hgnn_embeddings: Dict[str, torch.Tensor],
    factory: PersonalMLPFactory,
    alpha: float,
    sgd_steps: int,
    margin: float,
    batch_size: int,
    lifespan_days: int,
    cold_start_daily: bool = False,
    k: int = 10,
    ground_truth_t: int = 12,
) -> Dict:
    """Run one full simulation with given hyperparameters, return metrics."""
    backend = LocalDictBackend()
    ttl = lifespan_days * 24 * 3600

    lifecycle = MLPLifecycleManager(
        factory=factory,
        backend=backend,
        max_size=200,
        ttl_seconds=ttl,
        alpha=alpha,
    )

    adaptation = TripletAdaptation(
        sgd_steps=sgd_steps,
        learning_rate=1e-4,
        weight_decay=1e-6,
        margin=margin,
    )

    # Detect embedding dim: HGNN=64 → no MLP projection; CLIP=512 → project
    sample_emb = next(iter(hgnn_embeddings.values()))
    use_proj = sample_emb.shape[-1] == 512

    simulator = SessionSimulator(
        lifecycle_manager=lifecycle,
        clip_embeddings=hgnn_embeddings,
        adaptation=adaptation,
        trigger_every_n=batch_size,
        recommendation_k=k,
        ground_truth_t=ground_truth_t,
        use_mlp_projection=use_proj,
    )

    start = time.time()
    results = simulator.simulate_sessions(
        behavior_df.rename(columns={"customer_id": "customer_id", "t_dat": "t_dat"}),
        cold_start_daily=cold_start_daily,
    )
    elapsed = time.time() - start

    # Count interactions and adaptations
    total_interactions = sum(len(v) for v in results.values())
    total_adapted = sum(
        sum(1 for r in v if r.get("adapted", False))
        for v in results.values()
    )

    # Compute metrics from ground_truth in records
    # (SessionSimulator stores ground_truth per step)
    all_p, all_r, all_f1 = [], [], []
    for user_id, records in results.items():
        for rec in records:
            gt = [a for a in rec.get("ground_truth", []) if a in hgnn_embeddings]
            if not gt:
                continue
            # Use interaction_count as proxy — no recs stored in simulator,
            # so we evaluate coverage: how many gt articles are in embeddings
            # Real metric requires recommendations per step (added below)

    # Re-run with recommendation generation
    backend2 = LocalDictBackend()
    lifecycle2 = MLPLifecycleManager(
        factory=factory,
        backend=backend2,
        max_size=200,
        ttl_seconds=ttl,
        alpha=alpha,
    )
    from src.inference.recommender import PersonalizationEngine
    engine = PersonalizationEngine(
        lifecycle=lifecycle2,
        clip_embeddings=hgnn_embeddings,
        adaptation=TripletAdaptation(
            sgd_steps=sgd_steps,
            learning_rate=1e-4,
            margin=margin,
        ),
        trigger_every_n=batch_size,
        final_k=k,
    )

    # Simulate interactions and collect recommendations
    user_groups = behavior_df.sort_values("t_dat").groupby("customer_id")
    for user_id, user_txns in user_groups:
        articles = user_txns["article_id"].tolist()
        event_types = user_txns["event_type"].tolist()
        for i, (aid, etype) in enumerate(zip(articles, event_types)):
            if aid not in hgnn_embeddings:
                continue
            shown = [a for a in articles[max(0, i-10):i] if a != aid]
            engine.handle_interaction(
                user_id=str(user_id),
                article_id=aid,
                interaction_type=etype,
                shown_articles=shown,
            )
            # Get recommendations and compare to next ground_truth_t
            future = [a for a in articles[i+1:i+1+ground_truth_t] if a in hgnn_embeddings]
            if future:
                recs = engine.get_recommendations(str(user_id), aid, k=k)
                all_p.append(precision_at_k(recs, future, k))
                all_r.append(recall_at_k(recs, future, k))
                all_f1.append(f1_at_k(recs, future, k))

    n = len(all_p)
    scale = 1e4
    return {
        "alpha": alpha,
        "sgd_steps": sgd_steps,
        "margin": margin,
        "batch_size": batch_size,
        "lifespan_days": lifespan_days,
        "cold_start_daily": cold_start_daily,
        "precision_e4": round(sum(all_p) / n * scale, 2) if n > 0 else 0.0,
        "recall_e4": round(sum(all_r) / n * scale, 2) if n > 0 else 0.0,
        "f1_e4": round(sum(all_f1) / n * scale, 2) if n > 0 else 0.0,
        "n_evals": n,
        "elapsed_s": round(elapsed, 1),
        "cache_stats": lifecycle.get_stats(),
    }


# ============================================================
# Sensitivity sweep
# ============================================================

SWEEP_CONFIGS = {
    "alpha":       [0.1, 0.3, 0.5, 0.7, 1.0],
    "sgd_steps":   [1, 5, 20, 50],
    "margin":      [1.0, 100.0, float("inf")],
    "batch_size":  [1, 3, 5, 9],
    "lifespan_days": [1, 7, 14, 21],
}

# Best values from sensitivity sweep (local CPU, 200 users, fake_behavior.csv, CLIP 512-dim)
# Source of truth: configs/personalization.yaml
FIXED_DEFAULTS = {
    "alpha":         0.7,
    "sgd_steps":     1,
    "margin":        float("inf"),
    "batch_size":    1,
    "lifespan_days": 14,
}


def run_sensitivity(
    behavior_df: pd.DataFrame,
    hgnn_embeddings: Dict[str, torch.Tensor],
    factory: PersonalMLPFactory,
    output_path: str = "results/sensitivity.csv",
    max_users: int = 200,
    k: int = 10,
):
    """
    One-at-a-time sensitivity sweep: vary one hyperparameter while holding others fixed.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Limit users for speed
    top_users = behavior_df["customer_id"].value_counts().head(max_users).index
    df = behavior_df[behavior_df["customer_id"].isin(top_users)].copy()
    logger.info("Sensitivity sweep on %d users", df["customer_id"].nunique())

    rows = []

    for param, values in SWEEP_CONFIGS.items():
        logger.info("=== Sweeping %s over %s ===", param, values)
        for val in values:
            cfg = dict(FIXED_DEFAULTS)
            cfg[param] = val
            logger.info("  %s=%s ...", param, val)

            row = run_single_config(
                behavior_df=df,
                hgnn_embeddings=hgnn_embeddings,
                factory=factory,
                alpha=cfg["alpha"],
                sgd_steps=cfg["sgd_steps"],
                margin=cfg["margin"],
                batch_size=cfg["batch_size"],
                lifespan_days=cfg["lifespan_days"],
                k=k,
            )
            row["sweep_param"] = param
            rows.append(row)
            logger.info(
                "    P@%d=%.2f R@%d=%.2f F1@%d=%.2f (n=%d, %.1fs)",
                k, row["precision_e4"], k, row["recall_e4"],
                k, row["f1_e4"], row["n_evals"], row["elapsed_s"],
            )

    results_df = pd.DataFrame(rows)
    # Drop cache_stats dict column for CSV
    results_df = results_df.drop(columns=["cache_stats"], errors="ignore")
    results_df.to_csv(output_path, index=False)
    logger.info("Saved sensitivity results to %s", output_path)
    return results_df


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="M2 Hyperparameter Sensitivity Analysis")
    parser.add_argument("--behavior", default="data/fake_behavior.csv")
    parser.add_argument("--embeddings", default="src/models/article_embeddings_hgnn3.pt")
    parser.add_argument("--checkpoint", default="src/models/mlp_student.pt")
    parser.add_argument("--output", default="results/sensitivity.csv")
    parser.add_argument("--max-users", type=int, default=200,
                        help="Max users for sweep (default 200 for speed)")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    logger.info("Loading artifacts...")
    hgnn_embeddings = load_hgnn_embeddings(args.embeddings)
    behavior_df = load_behavior(args.behavior)
    factory = PersonalMLPFactory(args.checkpoint)

    logger.info("Starting sensitivity sweep (max_users=%d)...", args.max_users)
    results_df = run_sensitivity(
        behavior_df=behavior_df,
        hgnn_embeddings=hgnn_embeddings,
        factory=factory,
        output_path=args.output,
        max_users=args.max_users,
        k=args.k,
    )

    print("\n=== Sensitivity Results ===")
    for param in SWEEP_CONFIGS:
        print(f"\n--- {param} ---")
        sub = results_df[results_df["sweep_param"] == param][
            [param, "precision_e4", "recall_e4", "f1_e4", "n_evals"]
        ]
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
