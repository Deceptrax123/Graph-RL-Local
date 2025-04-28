from torch_geometric_temporal.nn import GConvGRU
from torch.nn import Module
import torch.nn.functional as F


class TemporalGNNModel(Module):
    def __init__(self, in_channels=3):
        super(TemporalGNNModel, self).__init__()

        self.l1 = GConvGRU(in_channels=in_channels, out_channels=128, K=3)
        self.l2 = GConvGRU(in_channels=128, out_channels=256, K=3)
        self.l3 = GConvGRU(in_channels=256, out_channels=3, K=3)

    def forward(self, x, edge_index):
        h = self.l1(x, edge_index)
        h = F.relu(h)
        h = self.l2(h, edge_index)
        h = F.relu(h)
        h = self.l3(h, edge_index)

        return h
