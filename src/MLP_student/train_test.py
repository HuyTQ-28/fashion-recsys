import sys
import os
import torch
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MLP_student.model import StudentMLP
from MLP_student.train import train_student_mlp
from MLP_student.build_dataset import DistillDataset  

def main():
    print("🚀 START TRAINING FUNCTION")

    # Load data
    data_path = "./data/article_embeddings_hgnn3.pt"
    clip_path = "./data/clip_embeddings.pt"
    config_path = "./configs/hgnn.yaml"
    output_path = "./data/mlp_student.pt"

    data = torch.load(data_path)
    clip_dict = torch.load(clip_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Loaded {len(clip_dict)} CLIP embeddings, {len(data['embeddings'])} HGNN embeddings")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = train_student_mlp(clip_dict, data, config, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)

    print(f"TRAINING FINISHED, model saved to {output_path}")

if __name__ == "__main__":
    main()