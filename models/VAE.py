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
from settings import vaeBias

import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from models.architectures.MLP import MLP_Encoder, MLP_Decoder
from models.architectures.CNN1D import CNN_Encoder, CNN_Decoder
from models.architectures.LSTM import LSTM_Encoder, LSTM_Decoder

from datasets import Vibration

arch = {
  'MLP':   {'enoder': MLP_Encoder, 'decoder': MLP_Decoder},
  'CNN1D': {'enoder': CNN_Encoder, 'decoder': CNN_Decoder},
  'LSTM':  {'enoder': LSTM_Encoder, 'decoder': LSTM_Decoder}
}

class VAE_Network(nn.Module):
    def __init__(self):
        super(VAE_Network, self).__init__()
        usedArch = arch[architechture]

        # Base encoder (whatever architecture you picked)
        # Suppose the encoder returns (hidden, activations)
        self.encoder = usedArch['enoder']()

        # Two linear layers to learn mu and logvar
        # The dimension here depends on how your encoder outputs its final embedding.
        self.fc_mu = nn.Linear(vaeBias,vaeBias)
        self.fc_logvar = nn.Linear(vaeBias,vaeBias)

        # Base decoder
        self.decoder = usedArch['decoder']()

    def encode(self, x):
        # If your chosen encoder returns hidden features and intermediate activations
        hidden, activations = self.encoder(x)

        # Derive mu and logvar from those hidden features
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar, activations

    @staticmethod
    def reparameterize(mu, logvar):
        # Reparameterization trick: z = mu + sigma * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Decode the latent vector z back to the original dimension
        return self.decoder(z)

    def forward(self, x):
        # Encode input to mu, logvar
        mu, logvar, activations = self.encode(x)

        # Sample z via reparameterization
        z = self.reparameterize(mu, logvar)

        # Decode to reconstruct
        reconstructed = self.decode(z)

        # Return everything needed for training
        return reconstructed, mu, logvar, activations

class LSTM_VAE(nn.Module):
    def __init__(self):
        super(LSTM_VAE, self).__init__()
        usedArch = arch[architechture]
        
        # 1) LSTM Encoder
        # Suppose LSTM_Encoder returns (x_enc, hidden_out)
        self.encoder = usedArch['enoder']()

        # 2) Two linear layers to learn mu, logvar
        #    You must match in_features to whatever your encoder outputs
        #    (e.g., hidden_dim).  For example, if LSTM_Encoder outputs
        #    shape [B, hidden_dim], do:
        hidden_dim = vaeBias  # <-- ADAPT to match your LSTM_Encoder
        self.fc_mu = nn.Linear(vaeBias, vaeBias)
        self.fc_logvar = nn.Linear(vaeBias, vaeBias)

        # 3) LSTM Decoder
        # Suppose LSTM_Decoder takes a latent vector [B, vaeBias] as input
        # or maybe you shape it into [B, seq_len, vaeBias]
        self.decoder = usedArch['decoder']()

    def encode(self, x):
        """
        LSTM_Encoder forward() should return a tuple:
            x_enc  -> final feature representation (e.g., last hidden state)
            hidden -> internal hidden/cell states (if needed)
        """
        x_enc, hidden_out = self.encoder(x)
        
        # x_enc might be [B, hidden_dim] or [B, seq_len, hidden_dim].
        # If it's [B, seq_len, hidden_dim], you may want to take only the last time step.
        
        # Derive mu and logvar from final encoded features
        mu = self.fc_mu(x_enc)
        logvar = self.fc_logvar(x_enc)

        # "activations" can be empty or track anything you'd like
        activations = []
        return mu, logvar, activations

    @staticmethod
    def reparameterize(mu, logvar):
        # z = mu + sigma * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Here, we pass the latent z to the LSTM_Decoder.  In many LSTM VAEs,
        you reshape z from [B, latent_dim] into [B, seq_len, latent_dim]
        (if your decoder expects a sequence).  Or you set it as the initial
        hidden state. This depends on your LSTM_Decoder code.
        """
        x_dec, last_h = self.decoder(z)
        return x_dec

    def forward(self, x):
        """
        Full forward pass for training:
          1) Encode input into (mu, logvar).
          2) Sample z using reparameterization.
          3) Decode z -> reconstruction.
        """
        mu, logvar, activations = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, activations


class VAE(object):
    def __init__(self, dataDir='', normalVersion=True, test=False, modelPath=''):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.isNormalData = normalVersion

        # Initialize dataset (same as AE)
        if not test:
            self._init_dataset(dataDir, test)
            self.train_loader = self.data.train_loader
            self.test_loader = self.data.test_loader

        # Use the new VAE network instead of the old AE network
        self.model = VAE_Network() if architechture != 'LSTM' else LSTM_VAE()
        
        # Adam optimizer, same as AE
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=scheduler_stepSize, gamma=scheduler_gamma)

        self.trainLosses = []
        self.testLosses = []

        if test:
            self.loadModel(modelPath)

    def _init_dataset(self, dir, test):
        from datasets import Vibration
        self.data = Vibration(dir=dir, normalVersion=self.isNormalData, test=test)

    def loss_function(self, reconstructed_x, x, mu, logvar):
        # VAE reconstruction + KL divergence
        reconstruction_loss = F.mse_loss(
            reconstructed_x,
            x,
            reduction='sum'
        )
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + KLD

    def train(self, epoch):
        self.model.train()
        train_loss = 0

        for batch_idx, data in enumerate(tqdm(self.train_loader)):
            # data is typically a tuple (x, label)
            data = data[0].clone().detach().to(self.device)

            self.optimizer.zero_grad()

            # Forward pass through VAE
            reconstructed, mu, logvar, _ = self.model(data)
                # Suppose 'reconstructed_x' is [B, timeStamps, channels].
                # Just do:
            if architechture == 'LSTM':
                reconstructed = reconstructed.transpose(1, 2)
                # Now shape is [B, channels, timeStamps].

            # Compute VAE loss
            loss = self.loss_function(reconstructed, data, mu, logvar)

            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        epochTrainLoss = train_loss / len(self.train_loader.dataset)
        print(f"====> Epoch: {epoch} Average loss: {epochTrainLoss:.4f}")
        self.trainLosses.append(epochTrainLoss)
        self.scheduler.step()

    def validate(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in self.test_loader:
                data = data[0].clone().detach().to(self.device)

                # Forward pass
                reconstructed, mu, logvar, _ = self.model(data)
                if architechture == 'LSTM':
                    reconstructed = reconstructed.transpose(1, 2)
                # Accumulate validation loss
                test_loss += self.loss_function(reconstructed, data, mu, logvar).item()

        epochTestLoss = test_loss / len(self.test_loader.dataset)
        print(f"====> Validation set loss: {epochTestLoss:.8f}")
        self.testLosses.append(epochTestLoss)

    def test(self, dir):
        # Load test data
        self._init_dataset(dir, True)
        self.test_loader = self.data.test_loader

        self.model.eval()
        testLosses = []
        with torch.no_grad():
            for data in self.test_loader:
                data = data[0].clone().detach().to(self.device)
                reconstructed, mu, logvar, _ = self.model(data)
                if architechture == 'LSTM':
                    reconstructed = reconstructed.transpose(1, 2)
                testLosses.append(self.loss_function(reconstructed, data, mu, logvar).item())

        avgTestLoss = sum(testLosses) / len(testLosses)
        print(f"====> Test set average loss: {avgTestLoss:.8f}")
        return testLosses

    def saveModel(self, path='checkpoints/vae_model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'architechture': architechture,
            # ... plus any other relevant hyperparams
            'trainLoss': self.trainLosses[-1] if self.trainLosses else None,
            'testLoss': self.testLosses[-1] if self.testLosses else None,
        }, path)

    def loadModel(self, modelPath='checkpoints/vae_model.pth'):
        checkpoint = torch.load(modelPath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def printLossResult(self):
        import matplotlib.pyplot as plt
        import numpy as np
        _x = np.arange(len(self.trainLosses))
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