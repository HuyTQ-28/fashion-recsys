import torch
import torch.nn.functional as F
from torch import optim
import logging

logger = logging.getLogger(__name__)


def train_student(model, loader, epochs=10, lr=1e-3, patience=3, device='cuda', use_wandb=False):
    """
    Train Student MLP model with embeddings CLIP -> HGNN.
    
    model: StudentMLP model
    loader: DataLoader returns (clip, hgnn)
    epochs: max epochs
    lr: learning rate
    patience: early stopping patience
    device: 'cuda' or 'cpu'
    use_wandb: log to wandb if True
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    if use_wandb:
        import wandb

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for clip, hgnn in loader:
            clip = clip.to(device)
            hgnn = hgnn.to(device)

            optimizer.zero_grad()

            # Predict and normalize embeddings
            pred = F.normalize(model(clip), dim=1)
            hgnn_norm = F.normalize(hgnn, dim=1)

            # Loss: alignment
            loss = F.mse_loss(pred, hgnn_norm.detach())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches if n_batches > 0 else float('inf')
        print(f"Epoch {epoch}: average alignment loss = {avg_loss:.6f}")

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

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    logger.info(f"Training complete. Best loss: {best_loss:.6f}")
    return model