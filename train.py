import argparse, os, sys
import numpy as np
import imageio
from scipy import ndimage

from settings import channels, norm_trainDataDir, abnorm_trainDataDir, autoencoderNormPath, autoencoderAbnormPath, epochs

import torch
from torchvision.utils import save_image

# from models.VAE import VAE
from models.AE import AE

from utils import get_interpolations

torch.manual_seed(42)

normalPar = (norm_trainDataDir, True, autoencoderNormPath)
abnormalPar = (abnorm_trainDataDir, False,autoencoderAbnormPath)

def train(parameters):
    ae = AE(parameters[0], parameters[1])
    
    for epoch in range(1, epochs + 1):
        ae.train(epoch)
        ae.validate()
    # ae.printLossResult()
    ae.saveModel(parameters[2])
    # ae.printLossResult()

def main():
    try:
        train(normalPar)        # train normal autoencoder
        train(abnormalPar)      # train abnoral autoencoder
        
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")

if __name__ == "__main__":  
    main()