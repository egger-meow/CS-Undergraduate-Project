import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from settings import channels, timeStamps, embeddingSize, dropout, layers

import numpy as  np 

class LSTM_Encoder(nn.Module):
    def __init__(self, input_size=(channels, timeStamps)):
        super(LSTM_Encoder, self).__init__()
        self.lstm_enc1 = nn.LSTM(
            input_size=input_size[0], 
            hidden_size=embeddingSize,
            num_layers=layers, 
            batch_first=True)
        
        self.lstm_enc2 = nn.LSTM(
            input_size=embeddingSize, 
            hidden_size=int(embeddingSize/2),
            num_layers=layers, 
            batch_first=True)
        

    def forward(self, x):
        x_trans = torch.transpose(x, 1, 2)
        # print(x_trans.shape)
        out, (last_h_state, last_c_state) = self.lstm_enc1(x_trans)
        out, (last_h_state, last_c_state) = self.lstm_enc2(out)
        
        x_enc = last_h_state.squeeze(dim=0)

        x_enc = x_enc.unsqueeze(1)

        x_enc = x_enc.repeat(1, timeStamps, 1)

        # print(out.shape)
        return x_enc, out

class LSTM_Decoder(nn.Module):
    def __init__(self, input_size=(channels, timeStamps)):
        super(LSTM_Decoder, self).__init__()
        self.act = nn.Sigmoid()
        
        self.lstm_dec1 = nn.LSTM(
            input_size=int(embeddingSize/2), 
            hidden_size=embeddingSize, 
            dropout=dropout, 
            num_layers=layers, 
            batch_first=True)
        
        self.lstm_dec2 = nn.LSTM(
            input_size=embeddingSize, 
            hidden_size=input_size[0], 
            dropout=dropout, 
            num_layers=layers, 
            batch_first=True)
        # self.fc = nn.Linear(hidden_size, input_size)
        self.linear = nn.Linear(embeddingSize, input_size[0])

    def forward(self, x):
        # print(x.shape)
        
        dec_out, (hidden_state, cell_state) = self.lstm_dec1(x)
        dec_out = self.linear(dec_out)
        # dec_out, (hidden_state, cell_state) = self.lstm_dec2(dec_out)
        # print(dec_out.shape)
        # if self.use_act:
        dec_out = self.act(dec_out)
            # pass
        dec_out_trans = torch.transpose(dec_out, 1, 2)
        # print(dec_out.shape)
        return dec_out, hidden_state
