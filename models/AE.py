import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np

from settings import epochs, cuda, channels, timeStamps, lr, scheduler_gamma, scheduler_stepSize, batchSize_aeNorm, batchSize_aeAbnorm
from settings import norm_trainDataDir, abnorm_trainDataDir
from settings import architechture, dataVerion, sampleRate, sampleRate_origin, slidingWindow_aeNorm, slidingWindow_aeAbnorm, stride, startChannel
from settings import embeddingSize, decoderShapeBias, dropout, layers


import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from models.architectures.MLP import MLP_Encoder, MLP_Decoder
from models.architectures.CNN1D import CNN_Encoder, CNN_Decoder
from models.architectures.LSTM import LSTM_Encoder, LSTM_Decoder

from datasets import Vibration

    
arch = {
    'MLP' : {'enoder': MLP_Encoder, 'decoder': MLP_Decoder},
    'CNN1D' : {'enoder': CNN_Encoder, 'decoder': CNN_Decoder},
    'LSTM' : {'enoder': LSTM_Encoder, 'decoder': LSTM_Decoder}
}

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        usedArch = arch[architechture]
        
        self.encoder = usedArch['enoder']()

        self.decoder = usedArch['decoder']()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    
class LSTMNetwork(nn.Module):
    def __init__(self):
        super(LSTMNetwork, self).__init__()
        usedArch = arch[architechture]
        
        self.encoder = usedArch['enoder']()
        self.decoder = usedArch['decoder']()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x_enc, enc_out = self.encoder(x)
        x_dec, last_h = self.decoder(x_enc)

        return x_dec

class AE(object):
    def __init__(self, dataDir = '', normalVersion = True, test = False, modelPath = ''):
        self.device = torch.device("cuda" if cuda else "cpu") 
        
        self.isNormalData = normalVersion
        if not test:
            self._init_dataset(dataDir, test)
            self.train_loader = self.data.train_loader
            self.test_loader = self.data.test_loader

        self.model = Network() if architechture != 'LSTM' else LSTMNetwork()
        
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=scheduler_stepSize, gamma=scheduler_gamma)
        
        self.trainLosses = []
        self.testLosses = []
        if test:
            self.loadModel(modelPath)
        
    def _init_dataset(self, dir, test):
        self.data = Vibration(dir = dir, normalVersion=self.isNormalData, test=test)

    def loss_function(self, reconstructed_x, x):
        # criterion = nn.SmoothL1Loss
        criterion = F.mse_loss
        # print(x.shape)
        # print(reconstructed_x.shape)
        
        loss = criterion(
            x.reshape(-1, channels * timeStamps),
            reconstructed_x.reshape(-1, channels * timeStamps), 
            reduction = 'sum')
        return loss

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        # print(self.train_loader[0])
        for batch_idx, data in enumerate(tqdm(self.train_loader)):
            data = data[0].clone().detach()
            
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch = self.model(data)

            loss = self.loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            
            log_interval = 10
            if False and batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))
        epochTrainLoss = train_loss / len(self.train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, epochTrainLoss))

        self.trainLosses.append(epochTrainLoss)
        self.scheduler.step()
        
    def validate(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                data = data[0].clone().detach()
                data = data.to(self.device)
                recon_batch = self.model(data)
                test_loss += self.loss_function(recon_batch, data).item()

        epochTestLoss = test_loss / len(self.test_loader.dataset)
        print('====> Validation set loss: {:.8f}'.format(epochTestLoss))
        self.testLosses.append(epochTestLoss)
        
    def printLossResult(self):
        _x = np.arange(0, epochs)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

        ax1.plot(_x, self.trainLosses)
        ax2.plot(_x, self.testLosses)
        ax1.title.set_text('training loss')
        ax2.title.set_text('validating loss')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('loss')
        plt.show()
        
    def saveModel(self, path = 'checkpoints/1003_ampSingle.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'architechture': architechture, 
            'dataVerion': dataVerion, 
            'sampleRate': sampleRate, 
            'sampleRate_origin': sampleRate_origin, 
            'slidingWindow_aeNorm': slidingWindow_aeNorm,  
            'slidingWindow_aeAbnorm': slidingWindow_aeAbnorm,  
            'stride': stride,            
            'startChannel': startChannel,   
            'channels': channels,  
            'timeStamps': timeStamps, 
            'embeddingSize': embeddingSize,  
            'decoderShapeBias': decoderShapeBias, 
            'dropout': dropout,  
            'layers': layers,  
            'trainLoss': self.trainLosses[-1],
            'testLoss': self.testLosses[-1],
        }, path)
        
    def test(self, dir):
        self._init_dataset(dir, True)
        self.test_loader = self.data.test_loader
        
        self.model.eval()
        testLosses = []
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                data = data[0].clone().detach()
                data = data.to(self.device)
                recon_batch = self.model(data)
                testLosses.append(self.loss_function(recon_batch, data).item())

        avgTestLoss = sum(testLosses) / len(testLosses)
        print('====> test set average loss: {:.8f}'.format(avgTestLoss))
        return testLosses
        
    def loadModel(self, modelPath = 'checkpoints/1003_ampSingle.pth'):

        checkpoint = torch.load(modelPath)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
