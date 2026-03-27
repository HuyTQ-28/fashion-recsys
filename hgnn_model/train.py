import os
import torch
import pandas as pd
from hgnn_model.model import HGNN
from hgnn_model.loss import contrastive_loss
from hgnn_model.build_graph import build_graph, build_edges_only
from hgnn_model.data_split import split_by_time


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    DATA_DIR = os.getenv("DATA_DIR", "../data")

    transactions_path = f"{DATA_DIR}/transactions_train.csv"
    clip_path = f"{DATA_DIR}/clip_embeddings.pt"
    save_path = f"{DATA_DIR}/article_embeddings2.pt"


    # 🔥 TRAIN GRAPH
    trans = pd.read_csv(transactions_path)
    trans["article_id"]=trans["article_id"].astype(str).str.zfill(10)
    edge_index, id2idx = build_graph(trans)

    # 🔥 VAL EDGES (không rebuild graph)
    # val_edge_index = build_edges_only(val_df, id2idx)

    # load CLIP
    clip_data = torch.load(clip_path)

    num_nodes = len(id2idx)
    X = torch.zeros(num_nodes, 512)

    for aid, idx in id2idx.items():
        if aid in clip_data:
            X[idx] = clip_data[aid]

    X = X.to(device)
    edge_index = edge_index.to(device)

    # if val_edge_index is not None:
    #     val_edge_index = val_edge_index.to(device)

    model = HGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 🔥 EARLY STOP
    # best_val_loss = float("inf")
    # patience = 5
    # counter = 0

    for epoch in range(80):
        # ===== TRAIN =====
        model.train()
        emb = model(X, edge_index)

        train_loss = contrastive_loss(emb, edge_index, num_nodes)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # ===== VALIDATION =====
        # model.eval()
        # with torch.no_grad():
        #     if val_edge_index is not None:
        #         val_loss = contrastive_loss(emb, val_edge_index, num_nodes)
        #         val_loss_val = val_loss.item()
        #     else:
        #         val_loss_val = None

        # print(f"Epoch {epoch} - Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss_val}")
        print(f"Epoch {epoch} - Train Loss: {train_loss.item():.4f}")

        # ===== EARLY STOP =====
        # if val_loss_val is not None:
        #     if val_loss_val < best_val_loss:
        #         best_val_loss = val_loss_val
        #         counter = 0

        #         # save best model
        #         torch.save({
        #             "embeddings": emb.detach().cpu(),
        #             "id2idx": id2idx
        #         }, save_path)

        #     else:
        #         counter += 1

        #     if counter >= patience:
        #         print("Early stopping triggered!")
        #         break

    # save split
    # train_df.to_csv(f"{DATA_DIR}/train_split.csv", index=False)
    # val_df.to_csv(f"{DATA_DIR}/val_split.csv", index=False)
    # test_df.to_csv(f"{DATA_DIR}/test_split.csv", index=False)