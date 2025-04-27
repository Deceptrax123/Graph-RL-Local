import torch
from torch_geometric.data import Data


def create_graph_data(joint_positions_at_t, skeletal_edge_index):

    node_features = joint_positions_at_t

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = skeletal_edge_index

    data = Data(x=x, edge_index=edge_index)

    return data
