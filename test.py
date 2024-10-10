import argparse, os, sys
import numpy as np
import imageio
from scipy import ndimage
import gc
import matplotlib.pyplot as plt
from settings import autoencoderNormPath, autoencoderAbnormPath, norm_testDataDir, abnorm_testDataDir

import torch
from torchvision.utils import save_image

# from models.VAE import VAE
from models.AE import AE


from utils import get_interpolations

torch.manual_seed(42)

def scatterTestReuslt(dataNoraml, dataAbnormal):
    x = np.arange(len(dataNoraml))

    # Plot array1 values with red dots
    plt.scatter(x, dataNoraml, color='red', label='normal data')

    # Plot array2 values with blue dots
    plt.scatter(x, dataAbnormal, color='blue', label='abnormal data')

    # Add a line to connect the dots (optional)
    plt.plot(x, dataNoraml, color='red', linestyle='', alpha=0.5)
    plt.plot(x, dataAbnormal, color='blue', linestyle='', alpha=0.2)

    # Label the axes and add a legend
    plt.xlabel('test sample ID')
    plt.ylabel('re-leveing score')
    plt.legend()

    # Show the plot
    plt.show()
    
if __name__ == "__main__":
    gc.collect()
    try:
        aeNormal = AE(test = True, modelPath = autoencoderNormPath)
        aeAbnormal = AE(test = True, modelPath = autoencoderAbnormPath)
        
        loss_aeNormal_dataNormal = aeNormal.test(norm_testDataDir)
        loss_aeAbnormal_dataNormal = aeAbnormal.test(norm_testDataDir)
        dataNormal_levelingScores = [x / y for x, y in zip(loss_aeNormal_dataNormal, loss_aeAbnormal_dataNormal)]
        print(loss_aeNormal_dataNormal)
        
        aeNormal = AE(test = True, modelPath = autoencoderNormPath)
        aeAbnormal = AE(test = True, modelPath = autoencoderAbnormPath)
        
        loss_aeNormal_dataAbnormal = aeNormal.test(abnorm_testDataDir)
        loss_aeAbnormal_dataAbnormal = aeAbnormal.test(abnorm_testDataDir)
        dataAbnormal_levelingScores = [x / y for x, y in zip(loss_aeNormal_dataAbnormal, loss_aeAbnormal_dataAbnormal)]
        print(loss_aeNormal_dataAbnormal)
        
        scatterTestReuslt(loss_aeNormal_dataNormal, loss_aeNormal_dataAbnormal)
        
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")        
    
