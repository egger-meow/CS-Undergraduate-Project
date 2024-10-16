import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from settings import channels, timeStamps, embeddingSize

import numpy as  np 

class LSTM_Encoder(nn.Module):
    def __init__(self, input_size=(channels, timeStamps)):
        super(LSTM_Encoder, self).__init__()
        pass

    def forward(self, x):
        pass

class LSTM_Decoder(nn.Module):
    def __init__(self, input_size=(channels, timeStamps)):
        super(LSTM_Decoder, self).__init__()
        pass

    def forward(self, x):
        pass
