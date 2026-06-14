import torch
import torch.nn as nn


class GATConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a1 = nn.Parameter(torch.zeros(out_features, 1))
        self.a2 = nn.Parameter(torch.zeros(out_features, 1))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        if len(x.shape) == 3:
            batch = x.shape[0]
            N = x.shape[2]
            
            x = x.permute(0, 2, 1).reshape(-1, self.in_features)
            Wh = torch.matmul(x, self.W)
            Wh = Wh.view(batch, N, self.out_features)
            
            # More efficient attention calculation
            e1 = torch.matmul(Wh, self.a1).squeeze(-1)
            e2 = torch.matmul(Wh, self.a2).squeeze(-1)
            e = e1.unsqueeze(2) + e2.unsqueeze(1)
            e = self.leakyrelu(e)
            
            zero_vec = -9e15 * torch.ones_like(e)
            adj_expanded = adj.unsqueeze(0).expand(batch, N, N)
            attention = torch.where(adj_expanded > 0, e, zero_vec)
            attention = torch.softmax(attention, dim=2)
            
            h_prime = torch.matmul(attention, Wh)
            h_prime = h_prime.permute(0, 2, 1)
            
            return h_prime
        else:
            x = x.transpose(0, 1)
            Wh = torch.matmul(x, self.W)
            
            N = Wh.size(0)
            
            # More efficient attention calculation
            e1 = torch.matmul(Wh, self.a1).squeeze(-1)
            e2 = torch.matmul(Wh, self.a2).squeeze(-1)
            e = e1.unsqueeze(1) + e2.unsqueeze(0)
            e = self.leakyrelu(e)

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = torch.softmax(attention, dim=1)

            h_prime = torch.matmul(attention, Wh)
            return h_prime.transpose(0, 1).unsqueeze(0)
