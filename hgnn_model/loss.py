import torch

def contrastive_loss(emb, edge_index, num_nodes, margin=0.5):
    import torch.nn.functional as F

    emb = F.normalize(emb, dim=1)

    src, pos = edge_index

    neg = torch.randint(0, num_nodes, (len(src),), device=emb.device)

    pos_dist = ((emb[src] - emb[pos])**2).sum(dim=1)
    neg_dist = ((emb[src] - emb[neg])**2).sum(dim=1)

    loss = torch.relu(pos_dist - neg_dist + margin).mean()

    return loss