import torch
from torch import nn
from torch.nn import Linear, LSTM, GRU
from Models.encoder import SpatialEncoder
import torch.nn.functional as F


class SpatioTemporalActorCritic(nn.Module):
    def __init__(self, num_markers, node_dim, embedding_dim, temporal_hidden_dim, window_size):
        super(SpatioTemporalActorCritic, self).__init__()
        self.num_markers = num_markers
        self.node_dim = node_dim
        self.spatial_hidden_dim = embedding_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.window_size = window_size

        self.spatial_encoder = SpatialEncoder()
        self.temporal_encoder = LSTM(input_size=self.spatial_hidden_dim, hidden_size=self.temporal_hidden_dim, num_layers=1,
                                     batch_first=True)

        self.actor_fc = Linear(self.temporal_hidden_dim,
                               self.temporal_hidden_dim//2)
        self.actor_mu = Linear(self.temporal_hidden_dim//2, self.num_markers*3)
        self.actor_sigma = Linear(
            self.temporal_hidden_dim//2, self.num_markers*3)

        self.critic_fc = Linear(
            in_features=self.temporal_hidden_dim, out_features=self.temporal_hidden_dim//2)
        self.critic_value = Linear(self.temporal_hidden_dim//2, 1)

    def forward(self, graph_list):
        spatial_embeddings = []

        for graph in graph_list:

            if graph.batch is None:
                graph.batch = torch.zeros(graph.x.size(
                    0), dtype=torch.long, device=graph.x.device)
            embedding = self.spatial_encoder.forward(
                graph.x, graph.edge_index, graph.batch)

            spatial_embeddings.append(embedding)

        sequence = torch.stack(spatial_embeddings, dim=0).permute(1, 0, 2)

        _, (hidden, _) = self.temporal_encoder(sequence)
        sequence_embedding = hidden.squeeze(0)

        actor_f = F.relu(self.actor_fc(sequence_embedding))
        mu = self.actor_mu(actor_f)
        std = torch.clamp(self.actor_sigma(actor_f), -20, 2)

        critic_f = F.relu(self.critic_fc(sequence_embedding))
        value = self.critic_value(critic_f)

        return mu, std, value
