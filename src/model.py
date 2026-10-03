import torch
from torch import nn

from formulas import symmetric_normalize_adjacency


class VanillaGCNLayer(nn.Module):
    """A single Kipf-Welling style graph convolution layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        output = a_norm @ x @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output


class VanillaGCN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.gcn1 = VanillaGCNLayer(in_features, hidden_features)
        self.gcn2 = VanillaGCNLayer(hidden_features, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        a_norm = symmetric_normalize_adjacency(adjacency)
        h = torch.relu(self.gcn1(x, a_norm))
        h = self.dropout(h)
        return self.gcn2(h, a_norm)
