import torch
import torch.nn.functional as F

def hgnn_contrastive_loss_margin(
    emb,
    edge_index,
    edge_weight,
    num_nodes,
    num_neg=1,
    margin=1.0
):
    src, pos = edge_index
    num_edges = src.shape[0]

    h_src = emb[src]
    h_pos = emb[pos]

    pos_dist = ((h_src - h_pos) ** 2).sum(dim=1)

    neg_idx = torch.randint(
        0, num_nodes,
        (num_edges, num_neg),
        device=emb.device
    )

    h_neg = emb[neg_idx]

    neg_dist = ((h_src.unsqueeze(1) - h_neg) ** 2).sum(dim=2)
    neg_dist = neg_dist.mean(dim=1)

    # 🔥 margin loss (ổn định hơn)
    loss = F.relu(pos_dist - neg_dist + margin)

    # apply edge weight
    loss = loss * edge_weight

    return loss.mean()

def bpr_loss(emb, edge_index, num_nodes):
    src, pos = edge_index  # src → pos (next item)

    # 🔥 query = embedding của item hiện tại
    query = emb[src]

    # positive score
    pos_score = (query * emb[pos]).sum(dim=1)

    # negative sampling
    neg = torch.randint(0, num_nodes, (len(src),), device=emb.device)

    neg_score = (query * emb[neg]).sum(dim=1)

    # 🔥 BPR loss
    loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

    return loss
