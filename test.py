import argparse, os, sys
import numpy as np
import imageio
from scipy import ndimage
import gc

from defi import autoencoderNormPath, autoencoderAbnormPath, norm_testDataDir, abnorm_testDataDir

import torch
from torchvision.utils import save_image

# from models.VAE import VAE
from models.AE import AE


from utils import get_interpolations

parser = argparse.ArgumentParser(
        description='Main function to call training for different AutoEncoders')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--model', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='Vibration', metavar='N',
                    help='Which dataset to use')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)


print(args.model)

if __name__ == "__main__":
    gc.collect()
    try:
        autoenc = None
    except KeyError:
        print('---------------------------------------------------------')
        print('Model architecture not supported. ', end='')
        print('Maybe you can implement it?')
        print('---------------------------------------------------------')
        sys.exit()
    
    try:
        aeNormal = AE(args, test = True, modelPath = autoencoderNormPath)
        aeAbnormal = AE(args, test = True, modelPath = autoencoderAbnormPath)
        
        loss_aeNormal_dataNormal = aeNormal.test(norm_testDataDir)
        loss_aeAbnormal_dataNormal = aeAbnormal.test(norm_testDataDir)
        dataNormal_levelingScores = [x / y for x, y in zip(loss_aeAbnormal_dataNormal, loss_aeNormal_dataNormal)]
        print(dataNormal_levelingScores)
        
        loss_aeNormal_dataAbnormal = aeNormal.test(abnorm_testDataDir)
        loss_aeAbnormal_dataAbnormal = aeAbnormal.test(abnorm_testDataDir)
        dataAbnormal_levelingScores = [x / y for x, y in zip(loss_aeAbnormal_dataAbnormal, loss_aeNormal_dataAbnormal)]
        print(dataAbnormal_levelingScores)
        
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")        
    
