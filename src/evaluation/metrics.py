"""
Evaluation Metrics.

Owner: Member 4 (Frontend & Evaluation)

Paper reference: Section 3
- After each user interaction, predict K articles
- Ground truth: next T articles the user will actually buy
- K=10, T=12 (common practice)
- Report: Precision@K, Recall@K, F1@K (multiplied by 10^4 for readability)
"""

import logging
from typing import List, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def precision_at_k(recommended: List[str], ground_truth: List[str], k: int = 10) -> float:
    """
    Precision@K: fraction of recommended items that are relevant.

    P@K = |recommended[:K] ∩ ground_truth| / K

    Args:
        recommended: List of recommended article IDs (ordered).
        ground_truth: List of ground truth article IDs (next T purchases).
        k: Number of top recommendations to consider.

    Returns:
        Precision score in [0, 1].
    """
    if k == 0:
        return 0.0

    rec_set = set(recommended[:k])
    gt_set = set(ground_truth)
    return len(rec_set & gt_set) / k


def recall_at_k(recommended: List[str], ground_truth: List[str], k: int = 10) -> float:
    """
    Recall@K: fraction of relevant items that are recommended.

    R@K = |recommended[:K] ∩ ground_truth| / |ground_truth|

    Args:
        recommended: List of recommended article IDs (ordered).
        ground_truth: List of ground truth article IDs.
        k: Number of top recommendations to consider.

    Returns:
        Recall score in [0, 1].
    """
    if not ground_truth:
        return 0.0

    rec_set = set(recommended[:k])
    gt_set = set(ground_truth)
    return len(rec_set & gt_set) / len(gt_set)


def f1_at_k(recommended: List[str], ground_truth: List[str], k: int = 10) -> float:
    """
    F1@K: harmonic mean of Precision@K and Recall@K.

    Args:
        recommended: List of recommended article IDs.
        ground_truth: List of ground truth article IDs.
        k: Number of top recommendations.

    Returns:
        F1 score in [0, 1].
    """
    p = precision_at_k(recommended, ground_truth, k)
    r = recall_at_k(recommended, ground_truth, k)

    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate_user_session(
    recommendations_per_step: List[List[str]],
    ground_truths_per_step: List[List[str]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Evaluate a single user's session.

    Args:
        recommendations_per_step: List of recommendation lists (one per interaction).
        ground_truths_per_step: List of ground truth lists (one per interaction).
        k: Number of recommendations.

    Returns:
        Dict with mean P@K, R@K, F1@K across all steps.
    """
    precisions = []
    recalls = []
    f1s = []

    for recs, gt in zip(recommendations_per_step, ground_truths_per_step):
        if gt:  # Only evaluate when there's ground truth
            precisions.append(precision_at_k(recs, gt, k))
            recalls.append(recall_at_k(recs, gt, k))
            f1s.append(f1_at_k(recs, gt, k))

    return {
        "precision_at_k": np.mean(precisions) if precisions else 0.0,
        "recall_at_k": np.mean(recalls) if recalls else 0.0,
        "f1_at_k": np.mean(f1s) if f1s else 0.0,
        "n_evaluated_steps": len(precisions),
    }


def aggregate_results(
    user_results: Dict[str, Dict[str, float]],
    scale_factor: int = 10000,
) -> Dict[str, float]:
    """
    Aggregate results across all users.

    Paper convention: all metrics multiplied by 10^4 for readability.

    Args:
        user_results: Dict of user_id -> metrics dict.
        scale_factor: Multiplier for readability (default: 10^4 per paper).

    Returns:
        Dict with mean ± std for each metric (scaled).
    """
    all_p = [r["precision_at_k"] for r in user_results.values()]
    all_r = [r["recall_at_k"] for r in user_results.values()]
    all_f1 = [r["f1_at_k"] for r in user_results.values()]

    return {
        "precision_mean": np.mean(all_p) * scale_factor,
        "precision_std": np.std(all_p) * scale_factor,
        "recall_mean": np.mean(all_r) * scale_factor,
        "recall_std": np.std(all_r) * scale_factor,
        "f1_mean": np.mean(all_f1) * scale_factor,
        "f1_std": np.std(all_f1) * scale_factor,
        "n_users": len(user_results),
    }
