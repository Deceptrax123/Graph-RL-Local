import torch
from torch_geometric.data import Data


def create_graph_data(node_features, skeletal_edge_index):
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = skeletal_edge_index

    data = Data(x=x, edge_index=edge_index)

    return data
