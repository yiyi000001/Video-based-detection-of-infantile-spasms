import torch
import torch.nn as nn

class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv3d(in_channels, out_channels, kernel_size=(ks, 1, 1), padding=(ks // 2, 0, 0)) for ks in kernel_sizes
        ])

    def forward(self, x):
        out = [conv(x) for conv in self.convs]
        return sum(out)

class TemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        n, c, t, v = x.shape
        x = x.view(n * v, c, t)
        attn_weights = self.attention(x)
        attn_weights = attn_weights.view(n, v, 1, t)
        attn_weights = attn_weights.permute(0, 2, 3, 1).contiguous()
        return x * attn_weights

class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A

class AdaptiveGraphConv(nn.Module):
    def __init__(self, num_nodes, kernel_size=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(1, self.kernel_size, kernel_size=(1, 1))

    def forward(self, x):
        n, c, t, v = x.size()
        x_mean = x.mean(dim=1).view(n, 1, t, v)
        A = self.conv(x_mean)
        A = A.view(self.kernel_size, n, t, v)
        return A.permute(1, 0, 2, 3)

class TGCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_nodes):
        super().__init__()
        self.t_conv = MultiScaleTemporalConv(in_channels, out_channels, kernel_sizes=[3, 5, 7])
        self.graph_conv = ConvTemporalGraphical(out_channels, out_channels, kernel_size)
        self.attention = TemporalAttention(out_channels)
        self.adaptive_graph_conv = AdaptiveGraphConv(num_nodes=num_nodes, kernel_size=kernel_size)
        self.fc = nn.Linear(out_channels * num_nodes * kernel_size, 1)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(out_channels * num_nodes * kernel_size)

    def forward(self, x, A):
        n, c, t, v = x.size()
        A = self.adaptive_graph_conv(x)
        x = self.t_conv(x.unsqueeze(-1)).squeeze(-1)
        x, A = self.graph_conv(x, A)
        x = self.attention(x)
        x = x.view(n, -1)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
