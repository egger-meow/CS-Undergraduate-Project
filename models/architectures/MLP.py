import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from settings import channels, timeStamps, embeddingSize, decoderShapeBias

import numpy as  np 

class MLP_Encoder(nn.Module):
    def __init__(self, input_size=(channels, timeStamps)):
        super(MLP_Encoder, self).__init__()

        self.input_size = input_size
        self.embedd = embeddingSize


        self.encoder = nn.Sequential(
            nn.Linear(timeStamps, embeddingSize),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(embeddingSize, embeddingSize//2),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(embeddingSize//2, embeddingSize//4),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(embeddingSize//4, embeddingSize//8),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)

class MLP_Decoder(nn.Module):
    def __init__(self, input_size=(channels, timeStamps)):
        super(MLP_Decoder, self).__init__()
        self.input_channels = channels
        self.input_timeStamps = timeStamps
        self.channel_mult = embeddingSize
        self.output_channels = channels

        self.decoder = nn.Sequential(
            
            nn.Linear(embeddingSize//8, embeddingSize//4),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            
            nn.Linear(embeddingSize//4, embeddingSize//2),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            
            nn.Linear(embeddingSize//2, embeddingSize),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            
            nn.Linear(embeddingSize, timeStamps),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        y = self.decoder(x)
        return y
