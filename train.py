import argparse, os, sys
import numpy as np
import imageio
from scipy import ndimage

from settings import channels, norm_trainDataDir, abnorm_trainDataDir, autoencoderNormPath, autoencoderAbnormPath, epochs, trainNormalAutoEncoder

import torch
from torchvision.utils import save_image

# from models.VAE import VAE
from models.AE import AE

from utils import get_interpolations

torch.manual_seed(42)

normalPar = (norm_trainDataDir, autoencoderNormPath)
abnormalPar = (abnorm_trainDataDir, autoencoderAbnormPath)

if __name__ == "__main__":
    try:
        parameters = normalPar if trainNormalAutoEncoder else abnormalPar
        
        ae = AE(parameters[0])
        for epoch in range(1, epochs + 1):
            ae.train(epoch)
            ae.validate()
        ae.printLossResult()
        ae.saveModel(parameters[1])
        
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")