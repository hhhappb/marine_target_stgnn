import torch
import torch.nn as nn


class GraphGenerator(nn.Module):
    def __init__(self, num_nodes, num_profiles):
        super(GraphGenerator, self).__init__()
        self.num_nodes = num_nodes
        self.num_profiles = num_profiles
        self.adj = self._create_adjacency_matrix()

    def _create_adjacency_matrix(self):
        adj = torch.zeros(self.num_nodes, self.num_nodes)
        for i in range(self.num_nodes):
            adj[i, i] = 1
            if i > 0:
                adj[i, i - 1] = 1
            if i < self.num_nodes - 1:
                adj[i, i + 1] = 1
        return adj

    def forward(self, F):
        graphs = []
        for p in range(self.num_profiles):
            graphs.append(self.adj.to(F.device))
        return graphs
