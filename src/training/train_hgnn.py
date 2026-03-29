"""
HGNN Teacher Training Loop.

Owner: Member 1 (Data & Graph Learning)

Paper reference: Section 2, Equations 2-3
- Neighbor sampling: num_neighbors=[8,8,8], batch_size=128
- Early stopping with patience=5
- Negative sampling: random pairs, |E-| = |E+|
- Loss weights gamma per relation type

Usage:
    python -m src.training.train_hgnn --config configs/hgnn.yaml --graph data/processed/graph_poc.pt
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.models.hgnn import HGNN, ContrastiveLoss

logger = logging.getLogger(__name__)


def sample_negative_edges(
    num_nodes: int,
    num_negatives: int,
    positive_edge_set: set,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample random negative edges (pairs not in the graph).

    Args:
        num_nodes: Total number of nodes.
        num_negatives: Number of negative edges to sample (= |E+|).
        positive_edge_set: Set of (src, dst) tuples for positive edges.
        device: Target device.

    Returns:
        Negative edge index [2, num_negatives].
    """
    neg_edges = []
    while len(neg_edges) < num_negatives:
        src = torch.randint(0, num_nodes, (num_negatives * 2,))
        dst = torch.randint(0, num_nodes, (num_negatives * 2,))

        for s, d in zip(src.tolist(), dst.tolist()):
            if s != d and (s, d) not in positive_edge_set:
                neg_edges.append([s, d])
                if len(neg_edges) >= num_negatives:
                    break

    return torch.tensor(neg_edges[:num_negatives], dtype=torch.long, device=device).t()


def train_hgnn(
    graph: HeteroData,
    config: dict,
    device: str = "cuda",
    use_wandb: bool = False,
) -> Tuple[HGNN, Dict[str, torch.Tensor]]:
    """
    Train the HGNN teacher model.

    Args:
        graph: PyG HeteroData with node features and edge indices.
        config: Training configuration from configs/hgnn.yaml.
        device: Device to train on.
        use_wandb: Whether to log to Weights & Biases.

    Returns:
        Tuple of (trained HGNN model, embeddings dict {article_id: Tensor[64]}).
    """
    hgnn_config = config["hgnn"]

    # Detect relation types from graph
    relation_types = []
    for edge_type in graph.edge_types:
        rel_name = edge_type[1]  # (src, rel, dst)
        relation_types.append(rel_name)

    if not relation_types:
        raise ValueError("Graph has no edges!")

    logger.info(f"Relation types: {relation_types}")

    # Initialize model
    model = HGNN(
        layer_dims=hgnn_config["structural_layers"],
        relation_types=relation_types,
        relation_aggr=hgnn_config.get("relation_agg", "mean"),
        activation=hgnn_config.get("activation", "relu"),
    ).to(device)

    # Loss
    gamma = hgnn_config.get("gamma", [1.0])
    margin = hgnn_config.get("contrastive_margin", 1.0)
    criterion = ContrastiveLoss(gamma=gamma, margin=margin)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=hgnn_config["learning_rate"],
        weight_decay=hgnn_config.get("weight_decay", 0),
    )

    # Training loop
    max_epochs = hgnn_config.get("max_epochs", 100)
    patience = hgnn_config.get("early_stopping_patience", 5)
    best_loss = float("inf")
    patience_counter = 0

    if use_wandb:
        import wandb

    node_features = graph["article"].x.to(device)
    num_nodes = node_features.size(0)

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        # Prepare edge data
        edge_indices = {}
        positive_edges = {}
        positive_weights = {}
        negative_edges = {}

        for edge_type in graph.edge_types:
            rel_name = edge_type[1]
            edge_index = graph[edge_type].edge_index.to(device)
            edge_weight = graph[edge_type].edge_weight.to(device)

            edge_indices[rel_name] = edge_index
            positive_edges[rel_name] = edge_index
            positive_weights[rel_name] = edge_weight

            # Build positive edge set for negative sampling
            pos_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
            neg_edge = sample_negative_edges(num_nodes, edge_index.size(1), pos_set, device)
            negative_edges[rel_name] = neg_edge

        # Forward pass
        embeddings = model(node_features, edge_indices)

        # Loss
        loss = criterion(embeddings, positive_edges, positive_weights, negative_edges)
        loss.backward()
        optimizer.step()

        # Logging
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}: loss={loss.item():.4f}")

        if use_wandb:
            wandb.log({"hgnn/train_loss": loss.item(), "epoch": epoch})

        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Extract final embeddings
    with torch.no_grad():
        edge_indices = {}
        for edge_type in graph.edge_types:
            rel_name = edge_type[1]
            edge_indices[rel_name] = graph[edge_type].edge_index.to(device)

        final_embeddings = model(node_features, edge_indices).cpu()

    # Create article_id -> embedding mapping
    embeddings_dict = {}
    article_ids = graph["article"].article_ids
    for idx, aid in enumerate(article_ids):
        embeddings_dict[aid] = final_embeddings[idx]

    logger.info(f"Training complete. Final loss: {best_loss:.4f}")
    return model, embeddings_dict


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train HGNN teacher model")
    parser.add_argument("--config", default="configs/hgnn.yaml")
    parser.add_argument("--graph", required=True, help="Path to graph .pt file")
    parser.add_argument("--output_model", default="checkpoints/hgnn.pt")
    parser.add_argument("--output_embeddings", default="data/processed/hgnn_embeddings.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    graph = torch.load(args.graph, weights_only=False)
    model, embeddings = train_hgnn(graph, config, device=args.device, use_wandb=args.wandb)

    import os
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(model.state_dict(), args.output_model)
    torch.save(embeddings, args.output_embeddings)
    logger.info(f"Model saved to {args.output_model}, embeddings to {args.output_embeddings}")
