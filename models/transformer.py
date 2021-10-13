import torch.nn as nn

from .common import FNetLayer, FFN


class TransformerLayer(nn.Module):
    def __init__(self,
                 channels,
                 dropout):
        super(TransformerLayer, self).__init__()
        self.f_net = FNetLayer(channels, dropout)
        self.ff = FFN(channels, dropout)

    def forward(self, x, x_mask):
        x += self.f_net(x)
        x += self.ff(x, x_mask)
        x *= x_mask
        return x


class Transformer(nn.Module):
    def __init__(self,
                 channels=192,
                 n_layers=6,
                 dropout=0.1):
        super(Transformer, self).__init__()

        self.layers = nn.ModuleList([
            TransformerLayer(
                channels=channels,
                dropout=dropout
            ) for _ in range(n_layers)
        ])

    def forward(self, x, x_mask):
        for layer in self.layers:
            x = layer(x, x_mask)
        return x
