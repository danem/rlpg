import torch
import numpy as np
import rlpg.utils as utils
import rlpg.models.common as common

class DQN (torch.nn.Module):
    def __init__ (self, img_w, img_h, fc_size, n_channels, n_actions):
        super().__init__()

        self.channels = np.array([5,10,2]) * n_channels
        self.convs = common.Conv2DBlock(
            img_w, img_h, n_channels,
            [
                (self.channels[0], 3, 1, 1),
                (self.channels[1], 3, 2, 1),
                (4, 3, 2, 1),
                # (self.channels[2], 3, 1, 1),
            ]
        )
        # self.mp = torch.nn.MaxPool2d(3)
        # last_dims = utils.conv_size2d(self.convs.last_dims[0], self.convs.last_dims[1], 3, 3)
        # self.last_size = last_dims[0] * last_dims[1] * self.channels[-1]

        self.linear = common.LinearBlock(
            [self.convs.last_size, fc_size, n_actions],
            torch.relu
        )
        
    
    def forward (self, x):
        x = self.convs(x)
        x = x.view(-1, self.convs.last_size)
        # x = self.mp(x).view(-1, self.last_size)
        x = self.linear(x)
        return x


class DQN1D (torch.nn.Module):
    def __init__ (self, observation_size, fc_size, n_actions):
        super().__init__()
        self.layers = common.LinearBlock([
            observation_size,
            fc_size,
            fc_size//2,
            n_actions
        ], torch.relu)

    def forward (self, x):
        return self.layers(x)
