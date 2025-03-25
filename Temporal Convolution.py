import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size[0],  # 确保 out_channels 是整数
            kernel_size=(kernel_size[0], 1),
            stride=(stride, 1)
        )

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size[0], kc // self.kernel_size[0], t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A
