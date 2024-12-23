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
from settings import architechture, dataVerion, sampleRate, sampleRate_origin, slidingWindow_aeNorm, slidingWindow_aeAbnorm, stride
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
        
        self.encoder = usedArch['enoder']()  # Encoder architecture
        self.decoder = usedArch['decoder']()  # Decoder architecture

    def encode(self, x):
        """Encodes the input and returns the latent space and activations."""
        z, activations = self.encoder(x)  # Encoder outputs latent representation and activations
        return z, activations

    def decode(self, z):
        """Decodes the latent space back to the original input shape."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass: Encode, then Decode."""
        z, activations = self.encode(x)  # Encode input
        reconstructed = self.decode(z)  # Decode latent representation
        return reconstructed, activations

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
        # Suppose the encoder returns (x_enc, enc_out)
        x_enc, enc_out = self.encoder(x)

        # Suppose decoder returns (x_dec, last_h)
        x_dec, last_h = self.decoder(x_enc)

        # Return x_dec and an empty list as the “activations”
        return x_dec, []

class AE:
    def __init__(self, dataDir='', normalVersion=True, test=False, modelPath=''):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        """Initialize dataset."""
        self.data = Vibration(dir=dir, normalVersion=self.isNormalData, test=test)

    def loss_function(self, reconstructed_x, x, activations, sparsity_param=0.05, sparsity_weight=0.01):
        """
        Computes the loss as a combination of reconstruction loss and sparsity loss.
        """
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(
            x.reshape(-1, channels * timeStamps),
            reconstructed_x.reshape(-1, channels * timeStamps),
            reduction='sum'
        )
    
        # Sparsity loss
        sparsity_loss = sum(torch.mean(torch.abs(a - sparsity_param)) for a in activations)

        # Total loss
        total_loss = reconstruction_loss + sparsity_weight * sparsity_loss
        return total_loss

    def train(self, epoch):
        """Training loop."""
        self.model.train()
        train_loss = 0

        for batch_idx, data in enumerate(tqdm(self.train_loader)):
            data = data[0].clone().detach().to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            reconstructed, activations = self.model(data)
            if architechture == 'LSTM':
                reconstructed = reconstructed.transpose(1, 2)
            # Compute loss
            loss = self.loss_function(reconstructed, data, activations)
            loss.backward()
            train_loss += loss.item()

            self.optimizer.step()

        # Average loss for the epoch
        epochTrainLoss = train_loss / len(self.train_loader.dataset)
        print(f"====> Epoch: {epoch} Average loss: {epochTrainLoss:.4f}")

        self.trainLosses.append(epochTrainLoss)
        self.scheduler.step()

    def validate(self):
        """Validation loop."""
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for data in self.test_loader:
                data = data[0].clone().detach().to(self.device)

                # Forward pass
                reconstructed, activations = self.model(data)
                if architechture == 'LSTM':
                    reconstructed = reconstructed.transpose(1, 2)
                # Compute loss
                test_loss += self.loss_function(reconstructed, data, activations).item()

        # Average validation loss
        epochTestLoss = test_loss / len(self.test_loader.dataset)
        print(f"====> Validation set loss: {epochTestLoss:.8f}")
        self.testLosses.append(epochTestLoss)

    def printLossResult(self):
        """Plot training and validation loss."""
        _x = np.arange(0, len(self.trainLosses))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

        ax1.plot(_x, self.trainLosses)
        ax2.plot(_x, self.testLosses)
        ax1.set_title('Training Loss')
        ax2.set_title('Validation Loss')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        plt.show()

    def saveModel(self, path='checkpoints/1003_ampSingle.pth'):
        """Save the model checkpoint."""
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
            'channels': channels,
            'timeStamps': timeStamps,
            'embeddingSize': embeddingSize,
            'decoderShapeBias': decoderShapeBias,
            'dropout': dropout,
            'layers': layers,
            'trainLoss': self.trainLosses[-1] if self.trainLosses else None,
            'testLoss': self.testLosses[-1] if self.testLosses else None,
        }, path)

    def test(self, dir):
        """Testing loop."""
        self._init_dataset(dir, True)
        self.test_loader = self.data.test_loader

        self.model.eval()
        testLosses = []

        with torch.no_grad():
            for data in self.test_loader:
                data = data[0].clone().detach().to(self.device)

                # Forward pass
                reconstructed, activations = self.model(data)

                # Compute loss
                testLosses.append(self.loss_function(reconstructed, data, activations).item())

        avgTestLoss = sum(testLosses) / len(testLosses)
        print(f"====> Test set average loss: {avgTestLoss:.8f}")
        return testLosses

    def loadModel(self, modelPath='checkpoints/1003_ampSingle.pth'):
        """Load a model checkpoint."""
        checkpoint = torch.load(modelPath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
