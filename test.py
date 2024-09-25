import argparse, os, sys
import numpy as np
import imageio
from scipy import ndimage

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

# vae = VAE(args)
ae = AE(args)
architectures = {'AE':  ae,
                 'VAE': ae}

print(args.model)

if __name__ == "__main__":
    try:
        autoenc = architectures[args.model]
    except KeyError:
        print('---------------------------------------------------------')
        print('Model architecture not supported. ', end='')
        print('Maybe you can implement it?')
        print('---------------------------------------------------------')
        sys.exit()
    
    try:
        path1 = 'E:/download/20240604/20240604/vibdata/20240604204639.csv'
        path2 = 'E:/download/20240607/20240607/vibdata/20240607161223.csv'
        
        print(path2)
        autoenc.testDataset(path2)
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")        
    
