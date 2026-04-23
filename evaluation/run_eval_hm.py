"""
So sánh tất cả baselines + Our System (HGNN+EMA).

Evaluation protocol (paper Section 3 + Appendix C): K=10, T=12, metrics ×10^4

Chạy:
    python -m evaluation.run_eval_hm
    python -m evaluation.run_eval_hm --sample 500
    python -m evaluation.run_eval_hm --output results/eval_hm.csv
    python -m evaluation.run_eval_hm --no-lightgcn --no-synthetic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.data_split import (
    load_eval_dataset, EvalDataset,
    HM_TRAIN_CSV, HM_FULL_CSV, HM_TRAIN_START, HM_TRAIN_END, HM_TEST_START, HM_TEST_END,
    PAPER_K, PAPER_T,
)
from evaluation.baselines import (
    RandomBaseline, PopularItems,
    LightGCNBaseline,
    HGNNNoEMA, HGNNWithEMAInProcess,
)
from src.metrics import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k, f1_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCALE = 10_000
SYNTHETIC_CSV = Path("data/subset_1week/synthetic_interactions.csv")


def compute_metrics(recommended: List[str], ground_truth: List[str], k: int) -> Dict[str, float]:
    return {
        f"HR@{k}":   hit_rate_at_k(recommended, ground_truth, k),
        f"NDCG@{k}": ndcg_at_k(recommended, ground_truth, k),
        f"P@{k}":    precision_at_k(recommended, ground_truth, k),
        f"R@{k}":    recall_at_k(recommended, ground_truth, k),
        f"F1@{k}":   f1_at_k(recommended, ground_truth, k),
    }


def evaluate_model(
    model_name: str,
    recommend_fn,
    dataset: EvalDataset,
    k: int,
) -> pd.DataFrame:
    rows = []
    users = list(dataset.ground_truth.keys())
    logger.info(f"[{model_name}] evaluating {len(users)} users, K={k}, T={PAPER_T}...")
    for user_id in users:
        gt   = dataset.ground_truth[user_id]
        recs = recommend_fn(user_id, k)
        m    = compute_metrics(recs, gt, k)
        m["user_id"] = user_id
        m["model"]   = model_name
        rows.append(m)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Aggregate per-user metrics → mean ± std, scale ×10^4 (paper convention)."""
    metric_cols = [f"HR@{k}", f"NDCG@{k}", f"P@{k}", f"R@{k}", f"F1@{k}"]
    rows = []
    for model_name, grp in df.groupby("model"):
        row = {"model": model_name}
        for col in metric_cols:
            vals = grp[col].values
            row[f"{col}_mean"] = float(np.mean(vals)) * SCALE
            row[f"{col}_std"]  = float(np.std(vals))  * SCALE
        row["n_users"] = len(grp)
        rows.append(row)
    return pd.DataFrame(rows)


def print_paper_table(agg: pd.DataFrame, k: int, model_order: List[str]) -> None:
    agg = agg.copy()
    agg["_order"] = agg["model"].map({m: i for i, m in enumerate(model_order)})
    agg = agg.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  EVALUATION RESULTS  (K={k}, T={PAPER_T})  —  metrics ×10^4  (paper Table 2/3)")
    print(sep)
    header = (f"{'Model':<28}  {'HR':>10}  {'NDCG':>10}  "
              f"{'Precision':>10}  {'Recall':>10}  {'F1-score':>10}")
    print(header)
    print("-" * 90)
    for _, row in agg.iterrows():
        hr   = f"{row[f'HR@{k}_mean']:.0f}±{row[f'HR@{k}_std']:.0f}"
        ndcg = f"{row[f'NDCG@{k}_mean']:.0f}±{row[f'NDCG@{k}_std']:.0f}"
        p    = f"{row[f'P@{k}_mean']:.0f}±{row[f'P@{k}_std']:.0f}"
        r    = f"{row[f'R@{k}_mean']:.0f}±{row[f'R@{k}_std']:.0f}"
        f1   = f"{row[f'F1@{k}_mean']:.0f}±{row[f'F1@{k}_std']:.0f}"
        print(f"  {row['model']:<26}  {hr:>10}  {ndcg:>10}  {p:>10}  {r:>10}  {f1:>10}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fashion RecSys evaluation (paper protocol)")
    parser.add_argument("--k",            type=int,  default=PAPER_K,
                        help=f"Number of recommendations (paper default: {PAPER_K})")
    parser.add_argument("--T",            type=int,  default=PAPER_T,
                        help=f"Ground truth window size (paper default: {PAPER_T})")
    parser.add_argument("--sample",       type=int,  default=None,
                        help="Sample N users for faster evaluation")
    parser.add_argument("--output",       type=str,  default="results/eval_hm_results.csv")
    parser.add_argument("--no-lightgcn",  action="store_true",
                        help="Skip LightGCN (slow training)")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Disable synthetic multi-interaction data for HGNN+EMA")
    args = parser.parse_args()

    K = args.k
    T = args.T

    logger.info("=== Step 1: Load data (temporal split) ===")
    for p, label in [(HM_TRAIN_CSV, "train subset"), (HM_FULL_CSV, "full CSV")]:
        if not p.exists():
            logger.error(f"{label} not found: {p.absolute()}")
            sys.exit(1)

    logger.info(f"Train: {HM_TRAIN_START}~{HM_TRAIN_END} (subset) | Test: {HM_TEST_START}~{HM_TEST_END} (full) | K={K} T={T}")
    dataset = load_eval_dataset(T=T, sample_users=args.sample)
    eval_users = set(dataset.ground_truth.keys())
    logger.info(f"Eval users: {len(eval_users)} | Train purchases: {len(dataset.train_purchases):,}")

    obs_rows = []
    for uid, aids in dataset.observed.items():
        for aid in aids:
            obs_rows.append({"customer_id": uid, "article_id": aid, "event_type": "purchase"})
    observed_df = pd.DataFrame(obs_rows)

    logger.info("=== Step 2: Fit baselines (paper Appendix C — EMA+KNN protocol) ===")

    logger.info("Fitting Random Baseline...")
    random_bl = RandomBaseline().fit(dataset.train_purchases, catalog=dataset.catalog)

    logger.info("Fitting Popular Items...")
    popular_bl = PopularItems().fit(dataset.train_purchases)

    lightgcn_bl = None
    if not args.no_lightgcn:
        logger.info("Fitting LightGCN (EMA+KNN) — may take a few minutes...")
        lightgcn_bl = LightGCNBaseline()
        lightgcn_bl.fit(dataset.train_purchases)
        lightgcn_bl.build_user_ema_vectors(observed_df)
    else:
        logger.info("Skipping LightGCN (--no-lightgcn)")

    logger.info("Loading HGNN No Personalization...")
    hgnn_no_ema = HGNNNoEMA().load()
    hgnn_no_ema.build_user_vectors(observed_df)

    synthetic_df = None
    if not args.no_synthetic and SYNTHETIC_CSV.exists():
        logger.info(f"Loading synthetic interactions: {SYNTHETIC_CSV}")
        synthetic_df = pd.read_csv(
            SYNTHETIC_CSV,
            usecols=["customer_id", "article_id", "event_type"],
            dtype={"customer_id": str, "article_id": str},
        )
        synthetic_df["article_id"] = synthetic_df["article_id"].str.zfill(10)
        synthetic_df = synthetic_df[synthetic_df["customer_id"].isin(eval_users)]
        logger.info(f"Synthetic interactions: {len(synthetic_df):,} rows, "
                    f"{synthetic_df['customer_id'].nunique():,} users")
    elif args.no_synthetic:
        logger.info("Synthetic interactions disabled (--no-synthetic)")
    else:
        logger.warning(f"Synthetic CSV not found: {SYNTHETIC_CSV} — running without")

    logger.info("Loading HGNN + EMA (Our System)...")
    hgnn_ema = HGNNWithEMAInProcess().load()
    logger.info(f"Adapting for {len(eval_users)} eval users ({len(observed_df):,} observed interactions)")
    hgnn_ema.build_and_adapt(observed_df, synthetic_interactions=synthetic_df)

    logger.info("=== Step 3: Inference + Scoring ===")
    all_dfs = []

    all_dfs.append(evaluate_model("Random Baseline",          lambda uid, k: random_bl.recommend(uid, k),   dataset, K))
    all_dfs.append(evaluate_model("Popular Items",            lambda uid, k: popular_bl.recommend(uid, k),  dataset, K))
    if lightgcn_bl is not None:
        all_dfs.append(evaluate_model("LightGCN",             lambda uid, k: lightgcn_bl.recommend(uid, k), dataset, K))
    all_dfs.append(evaluate_model("HGNN (No Personalization)", lambda uid, k: hgnn_no_ema.recommend(uid, k), dataset, K))
    all_dfs.append(evaluate_model("HGNN + EMA (Ours)",        lambda uid, k: hgnn_ema.recommend(uid, k),    dataset, K))

    per_user_df = pd.concat(all_dfs, ignore_index=True)
    agg_df = aggregate(per_user_df, K)

    model_order = [
        "Random Baseline",
        "Popular Items",
        "LightGCN",
        "HGNN (No Personalization)",
        "HGNN + EMA (Ours)",
    ]
    print_paper_table(agg_df, K, model_order)

    our_row = agg_df[agg_df["model"] == "HGNN + EMA (Ours)"]
    if not our_row.empty:
        our_f1 = float(our_row[f"F1@{K}_mean"].values[0])
        print("\n  F1-score improvement of HGNN+EMA (Ours) vs baselines:")
        for _, row in agg_df.iterrows():
            if row["model"] == "HGNN + EMA (Ours)":
                continue
            base_f1 = float(row[f"F1@{K}_mean"])
            if base_f1 > 0:
                improv = (our_f1 - base_f1) / base_f1 * 100
                print(f"    vs {row['model']:<28}: {improv:+.1f}%")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(out_path, index=False)
    per_user_df.to_csv(str(out_path).replace(".csv", "_per_user.csv"), index=False)
    logger.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
