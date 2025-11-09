import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureMap(nn.Module):
    """
    Shared-weight feature map layer (H1, H3)
    Mimics a small convolution with weight sharing across a map
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FeatureMap, self).__init__()
        # Weight sharing: one kernel per output channel applied to all input positions
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.kernel_size = kernel_size

    def forward(self, x):
        padding = self.kernel_size // 2
        x = F.conv2d(x, self.weight, bias=self.bias, stride=1, padding=padding)
        # Squashing function (tanh according to paper)
        x = torch.tanh(x)
        return x

class SubsamplingLayer(nn.Module):
    """
    Averaging/subsampling layer (H2, H4)
    Each unit averages over a small local region (2x2) and downsamples
    """
    def __init__(self, num_maps, pool_size):
        super(SubsamplingLayer, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        x = self.pool(x)
        return x
