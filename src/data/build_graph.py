"""
Co-Purchase Graph Construction.

Owner: Member 1 (Data & Graph Learning)

Builds an item-item co-occurrence graph from transaction data.
Supports single-relation (POC) and synthetic multi-relation edges (full).

Paper reference: Section 2 — Heterogeneous graph G where nodes = products,
edges = co-interactions with weight = number of users who interacted with both.

Usage:
    python -m src.data.build_graph --transactions data/subset/transactions.csv \
                                   --embeddings data/processed/clip_embeddings.pt \
                                   --output data/processed/graph_poc.pt
"""

import logging
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


def build_copurchase_pairs(
    transactions: pd.DataFrame,
    time_window_hours: Optional[float] = 24.0,
) -> Counter:
    """
    Build co-purchase pairs from transaction data.

    For each user, take all pairs of purchased articles within a time window.
    Edge weight = number of distinct users who co-purchased both items.

    Args:
        transactions: DataFrame with columns [customer_id, article_id, t_dat].
        time_window_hours: Time window in hours for co-purchase (None = all pairs per user).

    Returns:
        Counter mapping (article_id_1, article_id_2) -> co-purchase count.
    """
    pair_counts = Counter()

    grouped = transactions.groupby("customer_id")

    for customer_id, group in grouped:
        if time_window_hours is not None and "t_dat" in group.columns:
            # Group by time windows
            group = group.sort_values("t_dat")
            articles_in_window = []
            window_start = group.iloc[0]["t_dat"]

            for _, row in group.iterrows():
                if (row["t_dat"] - window_start).total_seconds() / 3600 > time_window_hours:
                    # Process current window
                    for a1, a2 in combinations(set(articles_in_window), 2):
                        pair = tuple(sorted([a1, a2]))
                        pair_counts[pair] += 1
                    articles_in_window = []
                    window_start = row["t_dat"]
                articles_in_window.append(row["article_id"])

            # Process last window
            for a1, a2 in combinations(set(articles_in_window), 2):
                pair = tuple(sorted([a1, a2]))
                pair_counts[pair] += 1
        else:
            # All pairs per user
            articles = group["article_id"].unique().tolist()
            for a1, a2 in combinations(articles, 2):
                pair = tuple(sorted([a1, a2]))
                pair_counts[pair] += 1

    logger.info(f"Built {len(pair_counts)} co-purchase pairs")
    return pair_counts


def build_graph(
    pair_counts: Counter,
    clip_embeddings: Dict[str, torch.Tensor],
    multi_relation: bool = False,
    light_threshold: int = 1,
    medium_threshold: int = 5,
) -> HeteroData:
    """
    Build a PyG HeteroData graph from co-purchase pairs.

    Args:
        pair_counts: Counter of (article_id_1, article_id_2) -> weight.
        clip_embeddings: Dict of article_id -> CLIP embedding tensor [512].
        multi_relation: If True, create synthetic multi-relation edges
            (light/medium/heavy co-purchase frequency buckets).
        light_threshold: Max weight for 'light' co-purchase.
        medium_threshold: Max weight for 'medium' co-purchase.

    Returns:
        PyG HeteroData object with node features and edge indices.
    """
    # Build article_id -> index mapping (only articles that appear in edges AND have embeddings)
    articles_in_graph = set()
    for (a1, a2) in pair_counts:
        if a1 in clip_embeddings and a2 in clip_embeddings:
            articles_in_graph.add(a1)
            articles_in_graph.add(a2)

    article_list = sorted(articles_in_graph)
    article_to_idx = {aid: idx for idx, aid in enumerate(article_list)}

    logger.info(f"Graph nodes: {len(article_list)}")

    # Build node features
    node_features = torch.stack([clip_embeddings[aid] for aid in article_list])

    # Build edges
    if multi_relation:
        # Synthetic multi-relation: light / medium / heavy co-purchase
        edge_types = {
            "light_copurchase": [],
            "medium_copurchase": [],
            "heavy_copurchase": [],
        }
        edge_weights = {k: [] for k in edge_types}

        for (a1, a2), weight in pair_counts.items():
            if a1 not in article_to_idx or a2 not in article_to_idx:
                continue

            idx1 = article_to_idx[a1]
            idx2 = article_to_idx[a2]

            if weight <= light_threshold:
                rel = "light_copurchase"
            elif weight <= medium_threshold:
                rel = "medium_copurchase"
            else:
                rel = "heavy_copurchase"

            edge_types[rel].append([idx1, idx2])
            edge_types[rel].append([idx2, idx1])  # undirected
            edge_weights[rel].extend([weight, weight])

        data = HeteroData()
        data["article"].x = node_features
        data["article"].article_ids = article_list

        for rel, edges in edge_types.items():
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data["article", rel, "article"].edge_index = edge_index
                data["article", rel, "article"].edge_weight = torch.tensor(
                    edge_weights[rel], dtype=torch.float
                )
                logger.info(f"  Edge type '{rel}': {edge_index.size(1)} edges")

    else:
        # Single relation: co-purchased
        edges = []
        weights = []

        for (a1, a2), weight in pair_counts.items():
            if a1 not in article_to_idx or a2 not in article_to_idx:
                continue

            idx1 = article_to_idx[a1]
            idx2 = article_to_idx[a2]

            edges.append([idx1, idx2])
            edges.append([idx2, idx1])  # undirected
            weights.extend([weight, weight])

        data = HeteroData()
        data["article"].x = node_features
        data["article"].article_ids = article_list

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            data["article", "copurchased", "article"].edge_index = edge_index
            data["article", "copurchased", "article"].edge_weight = torch.tensor(
                weights, dtype=torch.float
            )

        logger.info(f"  Total edges: {len(edges)}")

    # Store mapping for later use
    data.article_to_idx = article_to_idx
    data.idx_to_article = {v: k for k, v in article_to_idx.items()}

    return data


def build_graph_from_transactions(
    transactions_path: str,
    embeddings_path: str,
    multi_relation: bool = False,
    time_window_hours: float = 24.0,
) -> HeteroData:
    """
    End-to-end graph construction from transaction CSV and embeddings file.

    Args:
        transactions_path: Path to transactions CSV.
        embeddings_path: Path to CLIP embeddings .pt file.
        multi_relation: Whether to create synthetic multi-relation edges.
        time_window_hours: Time window for co-purchase pairs.

    Returns:
        PyG HeteroData graph.
    """
    from src.data.extract_embeddings import load_embeddings

    transactions = pd.read_csv(
        transactions_path,
        dtype={"article_id": str, "customer_id": str},
        parse_dates=["t_dat"],
    )

    clip_embeddings = load_embeddings(embeddings_path)

    pair_counts = build_copurchase_pairs(transactions, time_window_hours)
    graph = build_graph(pair_counts, clip_embeddings, multi_relation=multi_relation)

    return graph


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Build co-purchase graph")
    parser.add_argument("--transactions", required=True, help="Path to transactions CSV")
    parser.add_argument("--embeddings", required=True, help="Path to CLIP embeddings .pt")
    parser.add_argument("--output", required=True, help="Output path for graph .pt")
    parser.add_argument("--multi_relation", action="store_true", help="Create synthetic multi-relation edges")
    args = parser.parse_args()

    graph = build_graph_from_transactions(
        args.transactions, args.embeddings, multi_relation=args.multi_relation
    )
    torch.save(graph, args.output)
    logger.info(f"Graph saved to {args.output}")
