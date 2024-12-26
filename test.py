import argparse, os, sys
import numpy as np
import imageio
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
import gc
import matplotlib.pyplot as plt
from settings import autoencoderNormPath, autoEncoder, autoencoderAbnormPath, norm_testDataDir, abnorm_testDataDir, testFileNum
from math import sqrt
import torch
from torchvision.utils import save_image

from models.VAE import VAE
from models.AE import AE

from utils import get_interpolations

arch = {
  'AE':   AE,
  'VAE': VAE,
}

model = arch[autoEncoder]

torch.manual_seed(42)
def scatterTestReuslt2D(blue_x, blue_y, red_x, red_y):

    # Plotting the points
    plt.figure(figsize=(8, 6))
    plt.scatter(blue_x, blue_y, color='blue', label='normal')
    plt.scatter(red_x, red_y, color='red', label='abnormal')

    # Adding labels and title
    plt.xlabel("loss-normalAE")
    plt.ylabel("loss-abnormalAE")
    plt.title("conbine the two plots above")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
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
    aeNormal = model(test = True, normalVersion=True, modelPath = autoencoderNormPath)
    aeAbnormal = model(test = True, normalVersion=False, modelPath = autoencoderAbnormPath)
    
    loss_aeNormal_dataNormal        = aeNormal.test(norm_testDataDir)
    loss_aeAbnormal_dataNormal      = aeAbnormal.test(norm_testDataDir) 
    loss_aeNormal_dataAbnormal      = aeNormal.test(abnorm_testDataDir)
    loss_aeAbnormal_dataAbnormal    = aeAbnormal.test(abnorm_testDataDir) 
    
    loss_aeNormal_normalized    = normalization(loss_aeNormal_dataNormal + loss_aeNormal_dataAbnormal)
    loss_aeAbnormal_normalized  = normalization(loss_aeAbnormal_dataNormal + loss_aeAbnormal_dataAbnormal)
    
    loss_aeNormal_dataNormal        = loss_aeNormal_normalized[:testFileNum]
    loss_aeAbnormal_dataNormal      = loss_aeAbnormal_normalized[:testFileNum]
    
    loss_aeNormal_dataAbnormal        = loss_aeNormal_normalized[testFileNum:]
    loss_aeAbnormal_dataAbnormal      = loss_aeAbnormal_normalized[testFileNum:]
    
    z = zip(loss_aeNormal_dataNormal,loss_aeAbnormal_dataNormal)
    print(*z)
    dataNormal_levelingScores = [x / y for x, y in zip(loss_aeNormal_dataNormal, loss_aeAbnormal_dataNormal)]
    # dataNormal_levelingScores = [sqrt(i) for i in dataNormal_levelingScores]
    # print(len(loss_aeNormal_dataNormal), len(loss_aeAbnormal_dataNormal))
    
    dataAbnormal_levelingScores = [x / y for x, y in zip(loss_aeNormal_dataAbnormal, loss_aeAbnormal_dataAbnormal)]
    # dataAbnormal_levelingScores = [sqrt(i) for i in dataAbnormal_levelingScores]
    # print(len(loss_aeNormal_dataAbnormal), len(loss_aeAbnormal_dataAbnormal)) 
    
    # scatterTestReuslt(loss_aeNormal_dataNormal, loss_aeNormal_dataAbnormal, 'Single autoencoder(normal)')
    # scatterTestReuslt(loss_aeAbnormal_dataNormal, loss_aeAbnormal_dataAbnormal, 'Single autoencoder(abnormal)')
    scatterTestReuslt2D(loss_aeNormal_dataNormal, loss_aeAbnormal_dataNormal, loss_aeNormal_dataAbnormal, loss_aeAbnormal_dataAbnormal)
    scatterTestReuslt(dataNormal_levelingScores, dataAbnormal_levelingScores, 'double autoencoder')
    
def main():
    gc.collect()
    try:
        testSingleAEscore()
        
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")        
    
if __name__ == "__main__":
    main()