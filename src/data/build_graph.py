import argparse
import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVENT_TYPES = ['click', 'favorite', 'cart', 'purchase']

def build_hgnn_graph(df: pd.DataFrame, clip_dict: Dict[str, torch.Tensor], val_ratio: float = 0.1) -> HeteroData:
    """
    Constructs the multi-relational Item-Item graph for HGNN pre-training.
    Implements User-based split and Sparse Matrix Multiplication (A = R^T * R).
    """
    logger.info("Starting multi-relational graph construction...")

    df = df.copy()
    df['customer_id'] = df['customer_id'].astype(str)
    df['article_id'] = df['article_id'].astype(str).str.zfill(10)
    # 1. User-Based Split (Isolating users for validation)
    unique_users = df['customer_id'].unique()
    train_users, val_users = train_test_split(unique_users, test_size=val_ratio, random_state=42)
    
    # Filter interactions based on user splits
    df_train = df[df['customer_id'].isin(train_users)].copy()
    df_val = df[df['customer_id'].isin(val_users)].copy()
    
    logger.info(f"User split complete: {len(train_users)} train users, {len(val_users)} val users.")

    # 2. Global Item Mapping
    # The nodes correspond to products. We need a unified mapping for all items.
    unique_items = df['article_id'].unique()
    num_items = len(unique_items)
    item2idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
    
    # Map item IDs in both dataframes
    df_train['item_idx'] = df_train['article_id'].map(item2idx)
    df_val['item_idx'] = df_val['article_id'].map(item2idx)

    # 3. Build Node Features (CLIP Embeddings)
    # Serves as the initial label h_p^0
    embedding_dim = next(iter(clip_dict.values())).shape[-1] if clip_dict else 512
    x = torch.zeros((num_items, embedding_dim), dtype=torch.float)
    
    missing_emb_count = 0
    for item_id, idx in item2idx.items():
        key = str(item_id)
        if key in clip_dict:
            x[idx] = clip_dict[key].clone().detach().view(-1)
        else:
            missing_emb_count += 1
            
    if missing_emb_count > 0:
        logger.warning(f"Found {missing_emb_count} items missing from clip_dict. Initializing with zeros.")

    # Initialize the PyG HeteroData object
    data = HeteroData()
    data['item'].x = x

    # 4. Helper Function: Extract Item-Item Edges via Sparse Matrix Multiplication
    def extract_item_item_edges(df_split: pd.DataFrame, event_type: str):
        df_event = df_split[df_split['event_type'] == event_type]
        if df_event.empty:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)

        # Local user mapping to build the R matrix correctly
        local_users = df_event['customer_id'].unique()
        user2idx = {u: i for i, u in enumerate(local_users)}
        
        user_indices = df_event['customer_id'].map(user2idx).values
        item_indices = df_event['item_idx'].values
        num_local_users = len(local_users)
        
        # Build Sparse User-Item Matrix (R_c_i)
        ones = np.ones(len(df_event), dtype=np.float32)
        R_coo = sp.coo_matrix((ones, (user_indices, item_indices)), shape=(num_local_users, num_items))
        R_csr = R_coo.tocsr()
        
        # Binarize R_csr: We want to count distinct users, so multiple identical interactions 
        # from the same user should be capped at 1 before the dot product.
        R_csr.data = np.ones_like(R_csr.data)
        
        # Sparse Matrix Multiplication: A = R^T * R
        A_csr = R_csr.T.dot(R_csr)
        
        # Remove diagonal to prevent self-loops
        A_csr.setdiag(0)
        A_csr.eliminate_zeros()
        
        # Convert back to COO to extract edge structures
        A_coo = A_csr.tocoo()
        
        edge_index = torch.tensor(np.vstack((A_coo.row, A_coo.col)), dtype=torch.long)
        edge_weight = torch.tensor(A_coo.data, dtype=torch.float)
        
        return edge_index, edge_weight

    # 5. Populate the Heterogeneous Graph
    for event in EVENT_TYPES:
        edge_type = ('item', f'co_{event}', 'item')
        
        # Training Edges
        train_edge_index, train_edge_weight = extract_item_item_edges(df_train, event)
        data[edge_type].edge_index = train_edge_index
        data[edge_type].edge_weight = train_edge_weight # The a_{p,q}^{c_i} weight
        
        # Validation Edges (stored in the same object for streamlined PyG loading later)
        val_edge_index, val_edge_weight = extract_item_item_edges(df_val, event)
        data[edge_type].val_edge_index = val_edge_index
        data[edge_type].val_edge_weight = val_edge_weight
        
        logger.info(
            f"Relation {edge_type}: "
            f"{train_edge_index.size(1)} train edges, {val_edge_index.size(1)} val edges."
        )

    data['item'].id2idx = {str(k): v for k, v in item2idx.items()}

    return data

def main():
    parser = argparse.ArgumentParser(description="User-Split HeteroData Graph Builder")
    parser.add_argument("--interactions",    required=True,
                        help="Path to synthetic_interactions.csv")
    parser.add_argument("--clip_embeddings", required=True,
                        help="Path to clip_embeddings.pt dict")
    parser.add_argument("--output",          default="data/processed/graph_poc.pt",
                        help="Output PyG .pt file")
    parser.add_argument("--val_ratio",       type=float, default=0.1,
                        help="Fraction of users for validation (default 0.1)")
    args = parser.parse_args()

    logger.info("Loading interactions ...")
    df = pd.read_csv(args.interactions)

    logger.info("Loading CLIP embeddings ...")
    clip_dict = torch.load(args.clip_embeddings, weights_only=False)

    graph = build_hgnn_graph(df, clip_dict, val_ratio=args.val_ratio)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(graph, args.output)
    logger.info("Saved graph to %s", args.output)


if __name__ == "__main__":
    main()
