import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class HGNN(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128, out_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x