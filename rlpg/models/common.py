import rlpg.utils as utils 

import torch
import torch.nn as nn
from typing import List, Tuple

class Conv2DBlock (nn.Module):
    def __init__(
        self, 
        img_w: int,
        img_h: int,
        n_channels: int,
        layers: List[Tuple[int, int, int, int]]
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        last_sizes, last_channel = utils.conv_size2d(img_w, img_h, 3, 1, 1), n_channels

        for c, k, s, p in layers:
            conv = nn.Conv2d(last_channel, c, kernel_size=k, stride=s, padding=p)
            last_sizes = utils.conv_size2d(last_sizes[0], last_sizes[1], k, s, p)
            last_channel = c
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm2d(c))
        
        self.last_size = last_sizes[0] * last_sizes[1] * layers[-1][0]
        self.last_dims = last_sizes
    
    def forward (self, x):
        for conv, bn in zip(self.convs, self.batch_norms):
            x = torch.relu(conv(x))
            x = bn(x)
        return x

class LinearBlock (nn.Module):
    def __init__(self, layers: List[int], activation, initialization = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.out_size = layers[-1]
        self.in_size = last_sz = layers[0]

        for l in layers[1:]:
            layer = nn.Linear(last_sz, l)
            if initialization:
                layer.weight = initialization(layer.weight)

            self.layers.append(layer)
            last_sz = l

    
    def forward (self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)



