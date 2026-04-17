from src.models.hgnn import HGNN, ContrastiveLoss
import torch
import torch.nn as nn
import os
import yaml
import argparse
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

BASE_RELATIONS = ["click", "favorite", "cart", "purchase"]

def train_hgnn(
    graph: HeteroData, config: dict, device: str = "cuda", use_wandb: bool = False
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Trains the HGNN teacher model using Dynamic In-batch Negative Sampling.
    """
    layer_dims = config.get("layer_dims", [512, 256, 128, 64])
    epochs = config.get("max_epochs", 100)
    lr = config.get("learning_rate", 1e-4)
    weight_decay = config.get("weight_decay", 2e-5)
    batch_size = 2048
    num_neighbors = config.get("num_neighbors", [8, 8, 8])
    patience = config.get("patience", 5)

    model = HGNN(
        layer_dims=layer_dims,
        relation_types=BASE_RELATIONS,
        neighbor_aggr="mean",
        relation_aggr="mean",
    ).to(device)

    margin = config.get("margin", 2.0)
    
    # Initialize ContrastiveLoss WITH the margin to prevent unbounded negative loss
    criterion = ContrastiveLoss(margin=margin).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ==========================================
    # 1. SETUP NODE-BASED LOADERS
    # (Loaders now sample subgraphs without forcing edge labels)
    # ==========================================
    num_workers = 4
    logger.info("Initializing Subgraph Loaders...")
    train_loader = NeighborLoader(
        graph,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes='item',
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = NeighborLoader(
        graph,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes='item',
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if use_wandb:
        import wandb

    # ==========================================
    # 2. TRAINING & VALIDATION LOOP
    # ==========================================
    logger.info("Starting HGNN pre-training with In-batch Negative Sampling...")
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in tqdm(range(epochs)):
        # --- TRAIN PHASE ---
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            edge_index_dict = {}
            positive_edges, positive_weights, negative_edges = {}, {}, {}
            num_nodes_in_batch = batch['item'].num_nodes

            for rel in BASE_RELATIONS:
                edge_type = ('item', f'co_{rel}', 'item')
                
                # Check nếu đồ thị con này có chứa tương tác của relation này
                if edge_type in batch.edge_types and batch[edge_type].edge_index.size(1) > 0:
                    pos_edge = batch[edge_type].edge_index
                    raw_weights = batch[edge_type].edge_weight

                    # 1. Đưa cạnh vào Message Passing
                    edge_index_dict[rel] = pos_edge
                    
                    # 2. Đưa cạnh vào Positive Loss (Kèm Log-scale chống Gradient Explosion)
                    positive_edges[rel] = pos_edge
                    positive_weights[rel] = torch.log1p(raw_weights.to(torch.float32))

                    # 3. DYNAMIC NEGATIVE SAMPLING: Tự sinh cạnh âm bản trên GPU
                    neg_edge = negative_sampling(
                        edge_index=pos_edge,
                        num_nodes=num_nodes_in_batch,
                        num_neg_samples=pos_edge.size(1), # Tỷ lệ 1:1 theo paper
                        method='sparse'
                    )
                    negative_edges[rel] = neg_edge

            # Nếu sub-graph quá thưa, không có cạnh nào thì bỏ qua batch
            if not positive_edges:
                continue

            embeddings = model(batch['item'].x, edge_index_dict)
            loss = criterion(embeddings, positive_edges, positive_weights, negative_edges)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0.0
        total_global_nodes = graph['item'].num_nodes # Define global scope
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                val_edge_index_dict = {}
                val_pos_edges, val_pos_weights, val_neg_edges = {}, {}, {}
                num_nodes_in_batch = batch['item'].num_nodes

                # 1. Create a lightning-fast Global-to-Local mapping tensor on the GPU
                global_to_local = torch.full((total_global_nodes,), -1, dtype=torch.long, device=device)
                global_to_local[batch['item'].n_id] = torch.arange(num_nodes_in_batch, device=device)

                for rel in BASE_RELATIONS:
                    edge_type = ('item', f'co_{rel}', 'item')
                    
                    if edge_type in batch.edge_types:
                        # Message Passing using training history (Already localized by PyG)
                        if hasattr(batch[edge_type], 'edge_index') and batch[edge_type].edge_index.size(1) > 0:
                            val_edge_index_dict[rel] = batch[edge_type].edge_index
                        
                        # Calculating Loss using future validation edges
                        if hasattr(batch[edge_type], 'val_edge_index') and batch[edge_type].val_edge_index.size(1) > 0:
                            # 2. Translate Global IDs to Local Subgraph IDs
                            v_pos_edge_global = batch[edge_type].val_edge_index
                            v_pos_edge_local = global_to_local[v_pos_edge_global]
                            
                            # 3. Filter out edges where either the source or destination node is NOT in the current batch
                            valid_mask = (v_pos_edge_local[0] >= 0) & (v_pos_edge_local[1] >= 0)
                            
                            v_pos_edge = v_pos_edge_local[:, valid_mask]
                            v_raw_weights = batch[edge_type].val_edge_weight[valid_mask]

                            # Only proceed if there are actual validation edges in this localized subgraph
                            if v_pos_edge.size(1) > 0:
                                val_pos_edges[rel] = v_pos_edge
                                val_pos_weights[rel] = torch.log1p(v_raw_weights.to(torch.float32))

                                val_neg_edges[rel] = negative_sampling(
                                    edge_index=v_pos_edge,
                                    num_nodes=num_nodes_in_batch,
                                    num_neg_samples=v_pos_edge.size(1),
                                    method='sparse'
                                )

                # Skip the loss calculation if this specific batch subgraph has no validation edges
                if not val_pos_edges:
                    continue

                val_embeddings = model(batch['item'].x, val_edge_index_dict)
                v_loss = criterion(val_embeddings, val_pos_edges, val_pos_weights, val_neg_edges)
                total_val_loss += v_loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        
        logger.info(f"Epoch {epoch + 1:03d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if use_wandb:
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch + 1})

        # --- EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered! Best Val Loss: {best_val_loss:.4f}")
                break

    # ==========================================
    # 3. GENERATE FINAL EMBEDDINGS
    # ==========================================
    logger.info("Computing final graph embeddings from best model state...")
    if best_model_state:
        model.load_state_dict(best_model_state)
    model.eval()
    
    final_embeddings = torch.zeros((graph['item'].num_nodes, layer_dims[-1]), device='cpu')
    
    inf_loader = NeighborLoader(
        graph,
        num_neighbors=num_neighbors, 
        batch_size=1024,
        input_nodes='item'
    )

    with torch.no_grad():
        for batch in inf_loader:
            batch = batch.to(device)
            
            edge_index_dict = {}
            for rel in BASE_RELATIONS:
                edge_type = ('item', f'co_{rel}', 'item')
                if edge_type in batch.edge_types:
                    edge_index_dict[rel] = batch[edge_type].edge_index

            emb = model(batch['item'].x, edge_index_dict)
            
            batch_size_inf = batch['item'].batch_size
            global_nodes = batch['item'].n_id[:batch_size_inf]
            final_embeddings[global_nodes] = emb[:batch_size_inf].cpu()

    return model, final_embeddings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Train HGNN teacher model")
    parser.add_argument("--config", default="configs/hgnn.yaml")
    parser.add_argument("--graph", required=True, help="Path to graph .pt file")
    parser.add_argument("--output_model", default="checkpoints/hgnn.pt")
    parser.add_argument("--output_embeddings", default="data/processed/hgnn_embeddings.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loading graph from {args.graph}...")
    graph = torch.load(args.graph, weights_only=False)

    model, embeddings = train_hgnn(graph, config, device=args.device, use_wandb=args.wandb)

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_embeddings), exist_ok=True)

    output_data = {
        "embeddings": embeddings.cpu(),
        "id2idx": graph['item'].id2idx
    }

    torch.save(model.state_dict(), args.output_model)
    torch.save(output_data, args.output_embeddings)

    logger.info(f"Model saved to {args.output_model}")
    logger.info(f"Final embeddings saved to {args.output_embeddings}")