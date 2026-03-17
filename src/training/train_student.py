"""
Student MLP Knowledge Distillation Training.

Owner: Member 1 (Data & Graph Learning)

Paper reference: Section 2, Equation 4
- Alignment loss: MSE between MLP(h_CNN) and HGNN(h_CNN, E+)
- HGNN is frozen, only MLP weights update

Usage:
    python -m src.training.train_student --config configs/hgnn.yaml \
        --clip_embeddings data/processed/clip_embeddings.pt \
        --hgnn_embeddings data/processed/hgnn_embeddings.pt
"""

import logging
import os
from typing import Dict, Tuple

import torch
import torch.optim as optim

from src.models.student_mlp import StudentMLP, AlignmentLoss

logger = logging.getLogger(__name__)


def train_student_mlp(
    clip_embeddings: Dict[str, torch.Tensor],
    hgnn_embeddings: Dict[str, torch.Tensor],
    config: dict,
    device: str = "cuda",
    use_wandb: bool = False,
) -> StudentMLP:
    """
    Train the Student MLP via knowledge distillation from HGNN.

    Args:
        clip_embeddings: Dict of article_id -> CLIP embedding [512].
        hgnn_embeddings: Dict of article_id -> HGNN embedding [64] (teacher targets).
        config: Config dict with student_mlp section.
        device: Training device.
        use_wandb: Whether to log to WandB.

    Returns:
        Trained StudentMLP model.
    """
    mlp_config = config.get("student_mlp", config.get("hgnn", {}))

    # Align article sets (only articles that have both CLIP and HGNN embeddings)
    common_ids = sorted(set(clip_embeddings.keys()) & set(hgnn_embeddings.keys()))
    logger.info(f"Training on {len(common_ids)} articles with both CLIP and HGNN embeddings")

    if not common_ids:
        raise ValueError("No common article IDs between CLIP and HGNN embeddings!")

    # Prepare tensors
    X = torch.stack([clip_embeddings[aid] for aid in common_ids]).to(device)  # [N, 512]
    Y = torch.stack([hgnn_embeddings[aid] for aid in common_ids]).to(device)  # [N, 64]

    # Initialize model
    layers = mlp_config.get("layers", mlp_config.get("attribute_layers", [512, 256, 128, 64]))
    model = StudentMLP(layer_dims=layers).to(device)
    criterion = AlignmentLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=mlp_config.get("learning_rate", 1e-3),
        weight_decay=mlp_config.get("weight_decay", 0),
    )

    max_epochs = mlp_config.get("max_epochs", 100)
    patience = mlp_config.get("early_stopping_patience", 5)
    batch_size = 1024
    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    if use_wandb:
        import wandb

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Mini-batch training
        indices = torch.randperm(len(common_ids))
        for i in range(0, len(common_ids), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_x = X[batch_idx]
            batch_y = Y[batch_idx]

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}: alignment_loss={avg_loss:.6f}")

        if use_wandb:
            wandb.log({"student/alignment_loss": avg_loss, "epoch": epoch})

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    logger.info(f"Student MLP training complete. Best loss: {best_loss:.6f}")
    return model


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train Student MLP via distillation")
    parser.add_argument("--config", default="configs/hgnn.yaml")
    parser.add_argument("--clip_embeddings", required=True)
    parser.add_argument("--hgnn_embeddings", required=True)
    parser.add_argument("--output", default="checkpoints/student_mlp.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    clip_emb = torch.load(args.clip_embeddings, weights_only=False)
    hgnn_emb = torch.load(args.hgnn_embeddings, weights_only=False)

    model = train_student_mlp(clip_emb, hgnn_emb, config, device=args.device, use_wandb=args.wandb)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    logger.info(f"Student MLP saved to {args.output}")
