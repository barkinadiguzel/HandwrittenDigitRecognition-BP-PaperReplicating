import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import FeatureMap, SubsamplingLayer  

class BPNetwork(nn.Module):
    def __init__(self):
        super(BPNetwork, self).__init__()
        # Input: 16x16 normalized image, single channel
        # H1: shared-weight feature maps (4 maps, 24x24 units, 5x5 receptive field)
        self.h1 = FeatureMap(in_channels=1, out_channels=4, kernel_size=5) 
        
        # H2: averaging/subsampling layer (4 planes, 12x12 units, 2x2 subsampling)
        self.h2 = SubsamplingLayer(num_maps=4, pool_size=2)
        
        # H3: higher-level feature maps (12 maps, 8x8 units)
        self.h3 = FeatureMap(in_channels=4, out_channels=12, kernel_size=5)
        
        # H4: averaging/subsampling layer (12 maps, 4x4 units, 2x2 subsampling)
        self.h4 = SubsamplingLayer(num_maps=12, pool_size=2)
        
        # Fully connected output layer (10 classes)
        self.fc = nn.Linear(12*4*4, 10)

    def forward(self, x):
        x = self.h1(x)       # Feature extraction H1
        x = self.h2(x)       # Subsampling H2
        x = self.h3(x)       # Feature extraction H3
        x = self.h4(x)       # Subsampling H4
        x = x.view(x.size(0), -1)  # Flatten for FC
        x = self.fc(x)
        x = torch.tanh(x)    # According to paper: output +1 for correct class, -1 for others
        return x

if __name__ == "__main__":
    model = BPNetwork()
    x = torch.randn(1, 1, 16, 16)  # Dummy input
    y = model(x)
    print("Output shape:", y.shape)  # Should be [1, 10]
