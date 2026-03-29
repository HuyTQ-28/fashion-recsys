import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class HGNN(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=64):
        super().__init__()

        self.relations = ["click", "cart", "purchase"]

        self.convs1 = nn.ModuleDict({
            rel: SAGEConv(in_dim, hidden_dim) for rel in self.relations
        })

        self.convs2 = nn.ModuleDict({
            rel: SAGEConv(hidden_dim, out_dim) for rel in self.relations
        })

        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index_dict):
        h_list = []

        for rel in self.relations:
            if rel not in edge_index_dict:
                continue

            edge_index = edge_index_dict[rel]

            h = self.convs1[rel](x, edge_index)
            h = F.relu(h)
            h = self.dropout(h)

            h = self.convs2[rel](h, edge_index)

            h_list.append(h)

        # ===== aggregate relations (mean stable hơn attention) =====
        h = torch.stack(h_list, dim=0).mean(dim=0)

        return F.normalize(h, dim=1)