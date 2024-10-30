import argparse, os, sys
import numpy as np
import imageio
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
import gc
import matplotlib.pyplot as plt
from settings import autoencoderNormPath, autoencoderAbnormPath, norm_testDataDir, abnorm_testDataDir, testFileNum

import torch
from torchvision.utils import save_image

# from models.VAE import VAE
from models.AE import AE

from utils import get_interpolations

torch.manual_seed(42)

def scatterTestReuslt(dataNoraml, dataAbnormal, title):
    x = np.arange(len(dataNoraml))

    plt.title(title)
    # Plot array1 values with red dots
    plt.scatter(x, dataNoraml, color='blue', label='normal data')

    # Plot array2 values with blue dots
    plt.scatter(x, dataAbnormal, color='red', label='abnormal data')

    # Label the axes and add a legend
    plt.xlabel('test sample ID')
    plt.ylabel('re-leveing score')
    plt.legend()

    # Show the plot
    plt.show()

def normalization(data):
    scaler = MinMaxScaler()
    # Reshape to (-1, 1) to make it a 2D array for MinMaxScaler
    data_reshaped = [[x] for x in data]  
    normalized = scaler.fit_transform(data_reshaped)
    # Flatten the result back to a 1D list
    return [x[0] for x in normalized]

def interpolation(scores, newDim = testFileNum):
    data = np.array(scores)
    data = np.expand_dims(data, axis=0)
    # data is now shape (channels, timestamp)
    channels, timestamps = data.shape
    x_old = np.linspace(0, 1, timestamps)
    x_new = np.linspace(0, 1, newDim)
    
    # Interpolate for each channel
    interpolated_data = np.zeros((channels, newDim))
    for i in range(channels):
        interpolated_data[i] = np.interp(x_new, x_old, data[i])

    return interpolated_data[0].tolist()
    
def testSingleAEscore():
    aeNormal = AE(test = True, normalVersion=True, modelPath = autoencoderNormPath)
    aeAbnormal = AE(test = True, normalVersion=False, modelPath = autoencoderAbnormPath)
    
    loss_aeNormal_dataNormal        = interpolation(aeNormal.test(norm_testDataDir))
    loss_aeAbnormal_dataNormal      = interpolation(aeAbnormal.test(norm_testDataDir))
    loss_aeNormal_dataAbnormal      = interpolation(aeNormal.test(abnorm_testDataDir))
    loss_aeAbnormal_dataAbnormal    = interpolation(aeAbnormal.test(abnorm_testDataDir))
    
    loss_aeNormal_normalized    = normalization(loss_aeNormal_dataNormal + loss_aeNormal_dataAbnormal)
    loss_aeAbnormal_normalized  = normalization(loss_aeAbnormal_dataNormal + loss_aeAbnormal_dataAbnormal)
    
    loss_aeNormal_dataNormal        = loss_aeNormal_normalized[:testFileNum]
    loss_aeAbnormal_dataNormal      = loss_aeAbnormal_normalized[:testFileNum]
    
    loss_aeNormal_dataAbnormal        = loss_aeNormal_normalized[testFileNum:]
    loss_aeAbnormal_dataAbnormal      = loss_aeAbnormal_normalized[testFileNum:]
    
    
    dataNormal_levelingScores = [x / y for x, y in zip(loss_aeNormal_dataNormal, loss_aeAbnormal_dataNormal)]
    # print(len(loss_aeNormal_dataNormal), len(loss_aeAbnormal_dataNormal))
    
    dataAbnormal_levelingScores = [x / y for x, y in zip(loss_aeNormal_dataAbnormal, loss_aeAbnormal_dataAbnormal)]
    # print(len(loss_aeNormal_dataAbnormal), len(loss_aeAbnormal_dataAbnormal)) 
    
    scatterTestReuslt(loss_aeNormal_dataNormal, loss_aeNormal_dataAbnormal, 'Single autoencoder')
    
def main():
    gc.collect()
    try:
        test_normAEscoreDividedByAbnormAEscore()
        
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")        
    
if __name__ == "__main__":
    main()