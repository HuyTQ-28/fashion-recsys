import os
import torch
import pandas as pd

from hgnn_model.model import HGNN
from hgnn_model.loss import hgnn_contrastive_loss_margin
from hgnn_model.build_graph import build_graph_hgnn


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    DATA_DIR = os.getenv("DATA_DIR", "../data")

    data_path = f"{DATA_DIR}/fake_behavior.csv"
    clip_path = f"{DATA_DIR}/clip_embeddings.pt"
    save_path = f"{DATA_DIR}/article_embeddings_hgnn3.pt"

    # ===== LOAD DATA =====
    data = pd.read_csv(data_path)
    data["article_id"] = data["article_id"].astype(str).str.zfill(10)

    # ===== BUILD GRAPH =====
    edge_index_dict, edge_weight_dict, id2idx = build_graph_hgnn(data)
    for aid, idx in id2idx.items():
        print(f"aid_graph:{aid}")
        break

    # ===== LOAD CLIP EMB =====
    clip_data = torch.load(clip_path)
    for i, key in enumerate(list(clip_data.keys())[:1]):
        print(f"clip_key: {key}")

    num_nodes = len(id2idx)

    # 🔥 IMPORTANT: random init instead of zero
    X = torch.randn(num_nodes, 512) * 0.01

    for aid, idx in id2idx.items():
        if aid in clip_data:
            X[idx] = clip_data[aid]

    X = X.to(device)

    # ===== MOVE GRAPH TO DEVICE =====
    for rel in edge_index_dict:
        edge_index_dict[rel] = edge_index_dict[rel].to(device)
        edge_weight_dict[rel] = edge_weight_dict[rel].to(device)

    # ===== MODEL =====
    model = HGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    relation_weights = {
        "click": 1.0,
        "cart": 2.0,
        "purchase": 3.0
    }

    sample_size = 100_000  # 🔥 giảm cho ổn định hơn

    # ===== TRAIN LOOP =====
    for epoch in range(130):
        model.train()

        emb = model(X, edge_index_dict)
        # ===== DEBUG DISTANCE =====
        if epoch % 5 == 0:  # in mỗi 5 epoch cho đỡ spam
            with torch.no_grad():
                # lấy 1 relation bất kỳ (ví dụ purchase)
                rel = "purchase"
                edge_index = edge_index_dict[rel]

                num_edges = edge_index.shape[1]
                sample_size_debug = min(5000, num_edges)

                idx = torch.randperm(num_edges, device=device)[:sample_size_debug]
                edge_index_sample = edge_index[:, idx]

                src, pos = edge_index_sample

                pos_dist = ((emb[src] - emb[pos])**2).sum(dim=1).mean()

                neg_idx = torch.randint(
                    0, num_nodes,
                    (len(src),),
                    device=emb.device
                )
                neg_dist = ((emb[src] - emb[neg_idx])**2).sum(dim=1).mean()

                print(f"[DEBUG] pos_dist: {pos_dist:.4f}, neg_dist: {neg_dist:.4f}")

        if epoch == 0:
            print("Embedding norm:", emb.norm(dim=1).mean().item())

        total_loss = 0
        total_weight = 0

        for rel in edge_index_dict:
            edge_index = edge_index_dict[rel]
            edge_weight = edge_weight_dict[rel]

            num_edges = edge_index.shape[1]

            # ===== SAMPLE EDGES =====
            if num_edges > sample_size:
                idx = torch.randperm(num_edges, device=device)[:sample_size]
                edge_index_sample = edge_index[:, idx]
                edge_weight_sample = edge_weight[idx]
            else:
                edge_index_sample = edge_index
                edge_weight_sample = edge_weight

            loss = hgnn_contrastive_loss_margin(
                emb,
                edge_index_sample,
                edge_weight_sample,
                num_nodes,
                num_neg=10,
                margin=1.0
            )

            w = relation_weights.get(rel, 1.0)

            total_loss += w * loss
            total_weight += w

        total_loss = total_loss / total_weight

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 🔥 ổn định gradient
        optimizer.step()

        print(f"Epoch {epoch:02d} | Loss: {total_loss.item():.4f}")

    # ===== SAVE =====
    torch.save({
        "embeddings": emb.detach().cpu(),
        "id2idx": id2idx
    }, save_path)

    print(f"Training completed & embeddings saved in {save_path}")
