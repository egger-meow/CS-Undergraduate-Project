import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from settings import channels, timeStamps

import numpy as  np 

class CNN_Encoder(nn.Module):
    def __init__(self, input_size=(channels, timeStamps)):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size
        self.channel_mult = 32

        #convolutions
        self.conv = nn.Sequential(
            nn.Conv1d(
                    in_channels=input_size[0],
                    out_channels=self.channel_mult*1,
                    kernel_size=16,
                    stride=1,
                    padding=1),
            nn.BatchNorm1d(self.channel_mult*1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(
                self.channel_mult*1, 
                self.channel_mult*2, 
                kernel_size=8, 
                stride=2, 
                padding=1),
            nn.BatchNorm1d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(
                self.channel_mult*2, 
                self.channel_mult*4,
                kernel_size=4, 
                stride=2, 
                padding=1),
            nn.BatchNorm1d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(
                self.channel_mult*4, 
                self.channel_mult*4,
                kernel_size=4, 
                stride=2, 
                padding=1),
            nn.BatchNorm1d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),
            
        )

    def forward(self, x):
        return self.conv(x)

class CNN_Decoder(nn.Module):
    def __init__(self, input_size=(channels, timeStamps)):
        super(CNN_Decoder, self).__init__()
        self.input_channels = channels
        self.input_timeStamps = timeStamps
        self.channel_mult = 32
        self.output_channels = channels
        self.fc_output_dim = 512

        self.deconv = nn.Sequential(
            
            nn.ConvTranspose1d(
                self.channel_mult*4, 
                self.channel_mult*4,
                kernel_size=4, 
                stride=2, 
                padding=1),
            nn.BatchNorm1d(self.channel_mult*4),
            nn.ReLU(True),
            
            nn.ConvTranspose1d(
                self.channel_mult*4, 
                self.channel_mult*2,
                kernel_size=4, 
                stride=2, 
                padding=1),
            nn.BatchNorm1d(self.channel_mult*2),
            nn.ReLU(True),
            
            nn.ConvTranspose1d(
                self.channel_mult*2, 
                self.channel_mult*1,
                kernel_size=8, 
                stride=2, 
                padding=1),
            nn.BatchNorm1d(self.channel_mult*1),
            nn.ReLU(True),
            
            nn.ConvTranspose1d(
                self.channel_mult*1, 
                input_size[0],
                kernel_size=19, 
                stride=1, 
                padding=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        return self.deconv(x)
