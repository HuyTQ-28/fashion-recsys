"""
Chuẩn bị dữ liệu đánh giá theo paper protocol (Section 3 + Appendix C).

Evaluation protocol:
  - observed  = train week purchases (08-09→15) — EMA input
  - ground truth = test week purchases (08-16→22) — simulated future
  - Chỉ giữ users xuất hiện ở cả 2 tuần
  - Baselines fit trên toàn bộ train week để học item embeddings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd


HM_TRAIN_CSV = Path("data/subset_1week/transactions_train.csv")
HM_FULL_CSV  = Path("data/transactions_train_full.csv")

HM_TRAIN_START = "2020-08-09"
HM_TRAIN_END   = "2020-08-15"
HM_TEST_START  = "2020-08-16"
HM_TEST_END    = "2020-08-16"

PAPER_K = 20
PAPER_T = 12

CLIP_CHECKPOINT = Path("checkpoints/clip_embeddings.pt")


@dataclass
class EvalDataset:
    observed: Dict[str, List[str]] = field(default_factory=dict)
    ground_truth: Dict[str, List[str]] = field(default_factory=dict)
    train_purchases: pd.DataFrame = field(default_factory=pd.DataFrame)
    catalog: List[str] = field(default_factory=list)


def _load_known_articles(clip_checkpoint: Path = CLIP_CHECKPOINT) -> Set[str]:
    import torch
    data = torch.load(clip_checkpoint, map_location="cpu")
    return set(str(k) for k in data.keys())


def load_eval_dataset(
    train_csv: Path = HM_TRAIN_CSV,
    full_csv: Path = HM_FULL_CSV,
    clip_checkpoint: Path = CLIP_CHECKPOINT,
    T: int = PAPER_T,
    min_observed: int = 2,
    sample_users: int = None,
    seed: int = 42,
) -> EvalDataset:
    for p in [train_csv, full_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Khong tim thay: {p.absolute()}")

    rng = np.random.default_rng(seed)
    known_articles = _load_known_articles(clip_checkpoint)

    train_df = pd.read_csv(
        train_csv,
        usecols=["t_dat", "customer_id", "article_id"],
        parse_dates=["t_dat"],
    )
    train_df["article_id"]  = train_df["article_id"].astype(str).str.zfill(10)
    train_df["customer_id"] = train_df["customer_id"].astype(str)
    train_df = train_df[train_df["article_id"].isin(known_articles)].copy()
    train_df["event_type"] = "purchase"

    full_df = pd.read_csv(
        full_csv,
        usecols=["t_dat", "customer_id", "article_id"],
        parse_dates=["t_dat"],
    )
    full_df["article_id"]  = full_df["article_id"].astype(str).str.zfill(10)
    full_df["customer_id"] = full_df["customer_id"].astype(str)
    full_df = full_df[full_df["article_id"].isin(known_articles)]

    date_col = full_df["t_dat"].dt.date.astype(str)
    test_df = full_df[
        (date_col >= HM_TEST_START) & (date_col <= HM_TEST_END)
    ].copy()
    test_df["event_type"] = "purchase"

    common_users = set(train_df["customer_id"]) & set(test_df["customer_id"])

    observed: Dict[str, List[str]] = {}
    ground_truth: Dict[str, List[str]] = {}

    for user_id, group in train_df[train_df["customer_id"].isin(common_users)].groupby("customer_id"):
        group = group.sort_values("t_dat")
        seen: Dict[str, bool] = {}
        unique_items = []
        for aid in group["article_id"]:
            if aid not in seen:
                seen[aid] = True
                unique_items.append(aid)
        if len(unique_items) >= min_observed:
            observed[user_id] = unique_items

    for user_id, group in test_df[test_df["customer_id"].isin(common_users)].groupby("customer_id"):
        if user_id not in observed:
            continue
        group = group.sort_values("t_dat")
        seen = {}
        unique_items = []
        for aid in group["article_id"]:
            if aid not in seen:
                seen[aid] = True
                unique_items.append(aid)
        gt = unique_items[:T]
        if len(gt) >= 1:
            ground_truth[user_id] = gt

    observed = {u: v for u, v in observed.items() if u in ground_truth}

    if sample_users and len(ground_truth) > sample_users:
        sampled = rng.choice(list(ground_truth.keys()), size=sample_users, replace=False)
        observed     = {u: observed[u]     for u in sampled}
        ground_truth = {u: ground_truth[u] for u in sampled}

    return EvalDataset(
        observed=observed,
        ground_truth=ground_truth,
        train_purchases=train_df,
        catalog=list(known_articles),
    )


if __name__ == "__main__":
    dataset = load_eval_dataset()
    print(f"Eval users             : {len(dataset.ground_truth):,}")
    print(f"Train purchases (full) : {len(dataset.train_purchases):,}")

    obs_sizes = [len(v) for v in dataset.observed.values()]
    gt_sizes  = [len(v) for v in dataset.ground_truth.values()]
    import numpy as _np
    print(f"Observed per user      : mean={_np.mean(obs_sizes):.1f}  min={min(obs_sizes)}  max={max(obs_sizes)}")
    print(f"GT per user            : mean={_np.mean(gt_sizes):.1f}   min={min(gt_sizes)}   max={max(gt_sizes)}")

    if dataset.ground_truth:
        uid = next(iter(dataset.ground_truth))
        print(f"\nSample user : {uid[:16]}")
        print(f"  Observed (EMA input) : {dataset.observed[uid][:4]}...")
        print(f"  Ground truth (T<=12) : {dataset.ground_truth[uid]}")
