from torch_geometric.nn import ChebConv, global_add_pool
import torch.nn.functional as F
from torch.nn import Module

# Single time step (spatial)


class SpatialEncoder(Module):
    def __init__(self, init_out=16):
        super(SpatialEncoder, self).__init__()

        self.cheb1 = ChebConv(in_channels=3, out_channels=init_out, K=3)
        self.cheb2 = ChebConv(in_channels=init_out,
                              out_channels=init_out*2, K=3)
        self.cheb3 = ChebConv(in_channels=init_out*2,
                              out_channels=init_out*4, K=3)

    def forward(self, x, edge_index, batch):
        x = F.prelu(self.cheb1(x, edge_index=edge_index), weight=0.2)
        x = F.prelu(self.cheb2(x, edge_index=edge_index), weight=0.2)
        x = F.tanh(self.cheb3(x, edge_index=edge_index))  # Bounded

        x_agg = global_add_pool(x, batch=batch)

        return x_agg
