import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from architectures import  CNN_Encoder, CNN_Decoder
from datasets import Vibration

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.encoder = CNN_Encoder()

        self.decoder = CNN_Decoder()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class AE(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._init_dataset()
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader

        self.model = Network(args)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.03)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.9)
        
        self.trainLosses = []
        self.testLosses = []
        
    def _init_dataset(self):
        self.data = Vibration()

    def loss_function(self, recon_x, x):
        BCE = F.mse_loss(recon_x.view(-1, 600), x.view(-1, 600), reduction='sum')
        return BCE

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        # print(self.train_loader[0])
        for batch_idx, data in enumerate(tqdm(self.train_loader)):
            data = torch.tensor(data[0])
            
            data = data.to(self.device)
            # print(data.shape)
            self.optimizer.zero_grad()
            recon_batch = self.model(data)
            # print(recon_batch.shape)
            loss = self.loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            
            if False and batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))
        epochTrainLoss = train_loss / len(self.train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, epochTrainLoss))

        self.trainLosses.append(epochTrainLoss)
        self.scheduler.step()
        
    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                data = data[0].clone().detach()
                data = data.to(self.device)
                recon_batch = self.model(data)
                test_loss += self.loss_function(recon_batch, data).item()

        epochTestLoss = test_loss / len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(epochTestLoss))
        
        self.testLosses.append(epochTestLoss)
        
    def printLossResult(self):
        _x = np.arange(0, self.args.epochs )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

        ax1.plot(_x, self.trainLosses)
        ax2.plot(_x, self.testLosses)
        ax1.title.set_text('training loss')
        ax2.title.set_text('testing loss')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('loss')
        plt.show()
    def saveModel(self, path = 'checkpoints/0926.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.args.epochs,
            'trainLoss': self.trainLosses[-1],
            'testLoss': self.testLosses[-1],
        }, path)
        
    def testDataset(self, dataPath, modelPath = 'checkpoints/0926.pth'):
        self.data = Vibration(
            dataPath, 
            trainDataRatio=0)
        self.test_loader = self.data.test_loader
        checkpoint = torch.load(modelPath)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.test()