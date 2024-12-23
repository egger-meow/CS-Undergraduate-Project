import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from settings import channels, timeStamps, embeddingSize, decoderShapeBias, autoEncoder

import numpy as  np 
class CNN_Encoder(nn.Module):
    def __init__(self, input_size=(channels, timeStamps)):
        super(CNN_Encoder, self).__init__()
        self.input_size = input_size
        self.channel_mult = embeddingSize

        # Convolutions
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size[0],
                out_channels=self.channel_mult * 1,
                kernel_size=16,
                stride=4,
                padding=2,
            ),
            nn.BatchNorm1d(self.channel_mult * 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(
                self.channel_mult * 1,
                self.channel_mult * 2,
                kernel_size=8,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm1d(self.channel_mult * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(
                self.channel_mult * 2,
                self.channel_mult * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm1d(self.channel_mult * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(
                self.channel_mult * 4,
                self.channel_mult * 8,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm1d(self.channel_mult * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        activations = []  # Store activations for sparsity
        for layer in self.conv:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):  # Monitor activation layers for sparsity
                activations.append(x)

        return x, activations

class CNN_Decoder(nn.Module):
    def __init__(self, input_size=(channels, timeStamps)):
        super(CNN_Decoder, self).__init__()
        self.input_channels = channels
        self.input_timeStamps = timeStamps
        self.channel_mult = embeddingSize
        self.output_channels = channels

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(
                self.channel_mult * 8,
                self.channel_mult * 4,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm1d(self.channel_mult * 4),
            nn.ReLU(True),

            nn.ConvTranspose1d(
                self.channel_mult * 4,
                self.channel_mult * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm1d(self.channel_mult * 2),
            nn.ReLU(True),

            nn.ConvTranspose1d(
                self.channel_mult * 2,
                self.channel_mult * 1,
                kernel_size=8,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm1d(self.channel_mult * 1),
            nn.ReLU(True),

            nn.ConvTranspose1d(
                self.channel_mult * 1,
                input_size[0],
                kernel_size=16,
                stride=4,
                padding=2,
            ),
            nn.Linear(decoderShapeBias, timeStamps),

            nn.Sigmoid(),  # Output scaled to [0, 1]
        )

    def forward(self, x):
        y = self.deconv(x)
        return y
