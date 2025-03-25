import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils.tgcn import ConvTemporalGraphical
from nets.utils.graph import Graph
# from net.utils.ops import GraphUnet

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, act, drop_p):
        super(GraphConv, self).__init__()
        self.act = act
        self.drop_p = drop_p
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, g, h):
        assert h.size(1) == self.weight.size(0), f"Shape mismatch: h.size(1)={h.size(1)}, self.weight.size(0)={self.weight.size(0)}"
        h = torch.matmul(h, self.weight)
        h = self.act(h)

        return h
class GraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, alpha, concat=True):
        super(GraphAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.concat = concat

        # Initialize weight parameters
        self.W = None
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        N, C, T, V = h.size()
        if self.W is None or self.W.size(0) != C:
            self.W = nn.Parameter(torch.empty(size=(C, self.out_dim), device=h.device))
            nn.init.xavier_uniform_(self.W.data, gain=1.414)

        h = h.permute(0, 2, 3, 1).contiguous().view(N * T, V, C)
        Wh = torch.matmul(h, self.W).view(N, T, V, self.out_dim)

        e = torch.zeros(N, T, V, V, device=h.device)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e_v = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        for v in range(V):
            e[:, :, v] = e_v[:, :, v, :]

        zero_vec = -9e15 * torch.ones_like(e)
        attentions = []
        for i in range(adj.size(0)):
            adj_i = adj[i].unsqueeze(0).expand(N, T, V, V)
            attention_i = torch.where(adj_i > 0, e, zero_vec)
            attention_i = F.softmax(attention_i, dim=3)
            attentions.append(attention_i)

        attention = torch.stack(attentions, dim=0).sum(dim=0)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            h_prime = F.elu(h_prime)

        h_prime = h_prime.permute(0, 3, 1, 2).contiguous()
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N, T, V, C = Wh.size()
        Wh_repeated_in_chunks = Wh.repeat_interleave(V, dim=2)
        Wh_repeated_alternating = Wh.repeat(1, 1, V, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=3)
        return all_combinations_matrix.view(N, T, V, V, 2 * C)


class Model(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, edge_importance_weighting, **kwargs):
        super(Model, self).__init__()

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        T = 205  # Number of time steps
        V = 18  # Number of joints
        C = in_channels
        self.graph_unet = GraphUnet(ks=[0.8, 0.5, 0.2], in_dim=C * V, out_dim=64, act=nn.ReLU(), drop_p=0)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(C * V)

        self.GraphSTNet_networks = nn.ModuleList([
            GraphSTNet(64, 64, kernel_size, 1, **kwargs),  # Layer 1
            GraphSTNet(64, 64, kernel_size, 1, **kwargs),
            GraphSTNet(64, 64, kernel_size, 1, **kwargs)


        ])

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.fcn = nn.Conv2d(64, num_class, kernel_size=1)

        self.graph_attention = GraphAttention(64, 64, dropout=0, alpha=0.2)

    def forward(self, x):

        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)


        g = self.A.unsqueeze(0).repeat(N * M, 1, 1, 1)


        h = x.permute(0, 2, 3, 1).contiguous()
        h_flattened = h.view(N * M * T, V * C)


        hs = []
        for i in range(g.size(1)):
            hs.append(self.graph_unet(g[:, i, :, :], h_flattened))

        h = torch.stack(hs, dim=1)
        h = h.sum(dim=1)


        h = h.view(N * M, T, 64).permute(0, 2, 1).contiguous()
        h = h.view(N, M, 64, T).permute(0, 2, 3, 1).contiguous()


        for idx, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):

            h, A = gcn(h, self.A * importance)

            h = self.graph_attention(h, A)


        h = F.avg_pool2d(h, h.size()[2:])
        h = h.view(N, M, -1, 1, 1).mean(dim=1)


        h = self.fcn(h)
        h = h.view(h.size(0), -1)


        return h


class GraphUnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.act = act
        self.drop_p = drop_p
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.adjust_dims = nn.ModuleList()
        prev_dim = in_dim
        down_dims = []
        for k in ks:
            next_dim = int(prev_dim * k)
            self.down_gcns.append(GraphConv(prev_dim, next_dim, act, drop_p))
            down_dims.append(prev_dim)
            prev_dim = next_dim
        for k in reversed(ks):
            next_dim = int(prev_dim / k)
            self.up_gcns.append(GraphConv(prev_dim, next_dim, act, drop_p))
            down_dims.pop()
            prev_dim = next_dim
        self.proj = nn.Linear(86, out_dim)

    def forward(self, g, h):

        down_features = []
        for i, down_gcn in enumerate(self.down_gcns):
            h = down_gcn(g, h)

            down_features.append(h)
            h = F.dropout(h, self.drop_p, training=self.training)

        for i, up_gcn in enumerate(self.up_gcns):
            h = up_gcn(g, h)

            down_feature = down_features[-(i + 1)]

            if h.size(1) != down_feature.size(1):
                if h.size(1) > down_feature.size(1):
                    h = h[:, :down_feature.size(1), ...]
                else:
                    down_feature = down_feature[:, :h.size(1), ...]
            h = h + down_feature

            if i < len(self.up_gcns) - 1:
                adjust_dim = nn.Linear(h.size(1), self.up_gcns[i + 1].weight.size(0)).to(h.device)
                h = adjust_dim(h)

        if h.size(1) != self.proj.in_features:
            if h.size(1) > self.proj.in_features:
                h = h[:, :self.proj.in_features]
            else:
                self.proj = nn.Linear(h.size(1), self.proj.out_features).to(h.device)

        h = self.proj(h)

        return h




class GraphSTNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: torch.zeros_like(x)
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        # Debug: print input shape
        # print(f"Input shape: {x.shape}")

        res = self.residual(x)

        # Debug: print residual shape
        # print(f"Residual shape: {res.shape}")

        x, A = self.gcn(x, A)

        # Debug: print shape after GCN
        # print(f"Shape after GCN: {x.shape}")

        x = self.tcn(x)

        # Debug: print shape after TCN
        # print(f"Shape after TCN: {x.shape}")

        x = x + res

        x = self.relu(x)

        return x, A

