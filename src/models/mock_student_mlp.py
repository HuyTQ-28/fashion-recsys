# src/models/mock_student_mlp.py
import torch
import torch.nn as nn

class StudentMLP(nn.Module):
    """
    Mock Student MLP — cùng architecture với M1 sẽ train.
    Input:  512-dim CLIP embedding
    Output: 64-dim structural embedding
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_mock_checkpoint(path: str = "checkpoints/student_mlp.pt"):
    """Tạo random checkpoint để test — swap bằng real checkpoint của M1 sau."""
    import os
    os.makedirs("checkpoints", exist_ok=True)
    model = StudentMLP()
    torch.save(model.state_dict(), path)
    print(f"Mock checkpoint saved to {path}")


def create_mock_clip_embeddings(
    n_articles: int = 1000,
    path: str = "data/processed/clip_embeddings.pt"
):
    """Tạo mock CLIP embeddings — swap bằng real embeddings của M3 sau."""
    import os
    os.makedirs("data/processed", exist_ok=True)
    
    # Tạo fake article_ids
    article_ids = [f"article_{i:06d}" for i in range(n_articles)]
    embeddings = {
        aid: torch.randn(512)  # 512-dim CLIP, chưa normalize
        for aid in article_ids
    }
    torch.save(embeddings, path)
    print(f"Mock CLIP embeddings ({n_articles} articles) saved to {path}")
    return article_ids


if __name__ == "__main__":
    create_mock_checkpoint()
    create_mock_clip_embeddings()