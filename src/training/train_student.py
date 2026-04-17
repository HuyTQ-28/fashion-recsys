import logging
import os
from typing import Dict, Any

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.models.student_mlp import StudentMLP, AlignmentLoss

logger = logging.getLogger(__name__)

def train_student_mlp(
    clip_embeddings: Dict[str, torch.Tensor],
    hgnn_embeddings_data: Dict[str, Any],
    config: dict,
    device: str = "cuda",
    use_wandb: bool = False,
) -> StudentMLP:
    """
    Train the Student MLP via knowledge distillation with Validation split.
    """
    mlp_config = config.get("student_mlp", config.get("hgnn", {}))
    lr = float(mlp_config.get("learning_rate", 1e-3))
    epochs = mlp_config.get("max_epochs", 100)
    batch_size = mlp_config.get("batch_size", 1024)
    patience = mlp_config.get("patience", 10)
    weight_decay = float(mlp_config.get("weight_decay", 1e-5))

    # 1. Tách cấu trúc dữ liệu của HGNN
    hgnn_tensor = hgnn_embeddings_data["embeddings"]
    id2idx = hgnn_embeddings_data["id2idx"]

    # 2. Lọc common_ids chuẩn xác
    common_ids = [
        aid for aid in clip_embeddings 
        if aid in id2idx
    ]

    if len(common_ids) == 0:
        raise ValueError("No overlap between CLIP and HGNN embeddings!")

    train_ids, val_ids = train_test_split(common_ids, test_size=0.1, random_state=42)
    logger.info(f"Distillation: {len(train_ids)} train items, {len(val_ids)} val items")

    # 3. Trích xuất Tensor an toàn qua index mapping
    def get_tensors(ids):
        x = torch.stack([clip_embeddings[aid] for aid in ids])
        y = torch.stack([hgnn_tensor[id2idx[aid]] for aid in ids])
        return x.to(device), y.to(device)

    x_train, y_train = get_tensors(train_ids)
    x_val, y_val = get_tensors(val_ids)

    # 3. Setup Model, Loss, Optimizer
    model = StudentMLP(layer_dims=mlp_config.get("layer_dims", [512, 256, 128, 64])).to(device)
    criterion = AlignmentLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    # 4. Training Loop
    for epoch in tqdm(range(epochs)):
        model.train()

        permutation = torch.randperm(x_train.size(0))
        epoch_train_loss = 0.0
        
        for i in range(0, x_train.size(0), batch_size):
            optimizer.zero_grad()
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_x.size(0)

        avg_train_loss = epoch_train_loss / x_train.size(0)

        # --- VALIDATION PHASE ---
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            avg_val_loss = criterion(val_outputs, y_val).item()

        logger.info(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if use_wandb:
            import wandb
            wandb.log({"train_distill_loss": avg_train_loss, "val_distill_loss": avg_val_loss})

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered. Best Val Loss: {best_val_loss:.6f}")
                break

    if best_state:
        model.load_state_dict(best_state)

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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loading CLIP embeddings from {args.clip_embeddings}...")
    clip_emb = torch.load(args.clip_embeddings, weights_only=False)
    
    logger.info(f"Loading HGNN teacher targets from {args.hgnn_embeddings}...")
    hgnn_emb = torch.load(args.hgnn_embeddings, weights_only=False)

    model = train_student_mlp(clip_emb, hgnn_emb, config, device=args.device, use_wandb=args.wandb)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    logger.info(f"Student MLP saved successfully to {args.output}")