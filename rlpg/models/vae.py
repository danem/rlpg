import torch
import torch.nn as nn
import torch.nn.functional as tf
import torch.distributions
from typing import List, Tuple

import rlpg.models.common as common

class Decoder(nn.Module):
    def __init__(
        self, 
        latent_dims: int,
        hidden_size: int,
        out_size: int
    ):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_size)

    def forward(self, z):
        z = tf.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z


class VariationalEncoder(nn.Module):
    def __init__(
        self, 
        in_size: int,
        hidden_size: int,
        latent_dims: int
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_dims)
        self.linear3 = nn.Linear(hidden_size, latent_dims)

        self.register_buffer("mean", torch.tensor(0.))
        self.register_buffer("var", torch.tensor(1.))
        self._dist = None
        # self.kl = 0
    
    def _get_dist (self):
        # https://stackoverflow.com/questions/59179609/how-to-make-a-pytorch-distribution-on-gpu
        if self._dist:
            return self._dist
        self._dist = torch.distributions.Normal(self.mean, self.var)
        return self._dist

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = tf.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self._get_dist().sample(mu.shape)
        kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl

class VariationalAutoencoder(nn.Module):
    def __init__(
        self, 
        in_size: int,
        hidden_size: int,
        latent_dims: int,
        out_size: int
    ):
        super().__init__()
        self.in_size = in_size
        self.encoder = VariationalEncoder(in_size, hidden_size, latent_dims)
        self.decoder = Decoder(latent_dims, hidden_size, out_size)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class VariationalEncoder2D(nn.Module):
    def __init__(self, img_w: int, img_h: int, n_channels: int, hidden_size: int, latent_dims: int):
        super().__init__()

        layers = [
            (5, 3, 1, 1),
            (8, 3, 2, 1)
        ]
        self.conv = common.Conv2DBlock(img_w, img_h, n_channels, layers)
        self.encoder = VariationalEncoder(self.conv.last_size, hidden_size, latent_dims)
        self.channels = layers[-1][0]
        self.out_size = latent_dims
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.conv.last_size)
        return self.encoder(x)

class VariationalDecoder2D (nn.Module):
    def __init__(self, img_w: int, img_h: int, i_channels: int, o_channels: int, hidden_dims: int, latent_dims: int) -> None:
        super().__init__()
        layers = [
            (4, 3, 1, 1),
            (o_channels, 3, 1, 1)
        ]
        self.in_channels = i_channels
        self.out_channels = o_channels
        self.img_w = img_w
        self.img_h = img_h
        self.decoder = Decoder(latent_dims, hidden_dims, img_w*img_h*i_channels)
        self.conv = common.Conv2DBlock(img_w, img_h, i_channels, layers)
    
    def forward (self, x):
        x = self.decoder(x)
        x = x.view(-1, self.in_channels, self.img_h, self.img_w)
        return torch.relu(self.conv(x))

class VariationalAutoencoder2D(nn.Module):
    def __init__(
        self, 
        img_w: int, img_h: int,
        in_channels: int,
        hidden_size: int,
        latent_dims: int,
    ):
        super().__init__()

        self.encoder = VariationalEncoder2D(img_w, img_h, in_channels, hidden_size, latent_dims)
        self.decoder = Decoder(latent_dims, hidden_size, in_channels * img_w * img_h)
        self.img_w = img_w
        self.img_h = img_h
        self.out_channels = in_channels

    def forward(self, x):
        z, kl = self.encoder(x)
        z = self.decoder(z)
        return z.view(-1, self.out_channels, self.img_h, self.img_w), kl


class QVAE2D (nn.Module):
    def __init__(
        self, 
        encoder: VariationalEncoder2D, # pretrained encoder
        hidden_size: int,
        n_actions: int
    ) -> None:
        super().__init__()

        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.layer1 = nn.Linear(self.encoder.out_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, n_actions)

    def forward (self, x):
        x = self.encoder(x)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return x
