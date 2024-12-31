import argparse
import os, sys
import numpy as np
import gc
import matplotlib.pyplot as plt
import torch

# ========== Your Imports from Both Scripts ========== #
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
from math import sqrt

# If you use them:
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import ...
# etc.

from settings import (
    channels, autoEncoder, norm_trainDataDir, abnorm_trainDataDir, 
    autoencoderNormPath, autoencoderAbnormPath, epochs,
    norm_testDataDir, abnorm_testDataDir, testFileNum
)

from models.VAE import VAE
from models.AE import AE

from utils import get_interpolations

# ========== Model Dictionary ========== #
arch = {
    'AE':  AE,
    'VAE': VAE,
}

model = arch[autoEncoder]

torch.manual_seed(42)

# ========== TRAINING PHASE (from train_II.py) ========== #

def train_phase():
    """
    Train both normal AE and abnormal AE in sequence.
    Equivalent to what you had in train_II.py
    """
    print("=== TRAIN PHASE: Starting training of normal & abnormal AEs ===")
    normalPar = (norm_trainDataDir, True, autoencoderNormPath)
    abnormalPar = (abnorm_trainDataDir, False, autoencoderAbnormPath)

    def train_autoencoder(parameters):
        # parameters: (dataDir, isNormal, modelPath)
        ae = model(parameters[0], parameters[1])
        
        for epoch in range(1, epochs + 1):
            ae.train(epoch)
            ae.validate()
            
        ae.printLossResult()
        ae.saveModel(parameters[2])

    # Train normal autoencoder
    train_autoencoder(normalPar)

    # Train abnormal autoencoder
    train_autoencoder(abnormalPar)
    print("=== TRAIN PHASE: Completed ===\n")

# ========== TESTING PHASE (from test_I.py) ========== #

def scatterTestReuslt2D(blue_x, blue_y, red_x, red_y):
    plt.figure(figsize=(8, 6))
    plt.scatter(blue_x, blue_y, color='blue', label='normal')
    plt.scatter(red_x, red_y, color='red', label='abnormal')
    plt.xlabel("loss-normalAE")
    plt.ylabel("loss-abnormalAE")
    plt.title("combine the two plots above")
    plt.legend()
    plt.grid(True)
    plt.show()

def scatterTestReuslt(dataNoraml, dataAbnormal, title):
    x = np.arange(len(dataNoraml))
    plt.title(title)
    plt.scatter(x, dataNoraml, color='blue', label='normal data')
    plt.scatter(x, dataAbnormal, color='red', label='abnormal data')
    plt.xlabel('test sample ID')
    plt.ylabel('re-leveling score')
    plt.legend()
    plt.show()

def normalization(data):
    scaler = MinMaxScaler()
    data_reshaped = [[x] for x in data]
    normalized = scaler.fit_transform(data_reshaped)
    return [x[0] for x in normalized]

def testSingleAEscore():
    """Equivalent to testSingleAEscore() from test_I.py."""
    aeNormal = model(test=True, normalVersion=True,  modelPath=autoencoderNormPath)
    aeAbnormal = model(test=True, normalVersion=False, modelPath=autoencoderAbnormPath)
    
    # Evaluate on normal test set
    loss_aeNormal_dataNormal     = aeNormal.test(norm_testDataDir)
    loss_aeAbnormal_dataNormal   = aeAbnormal.test(norm_testDataDir)
    # Evaluate on abnormal test set
    loss_aeNormal_dataAbnormal   = aeNormal.test(abnorm_testDataDir)
    loss_aeAbnormal_dataAbnormal = aeAbnormal.test(abnorm_testDataDir)
    
    # Combine & normalize
    loss_aeNormal_normalized   = normalization(loss_aeNormal_dataNormal + loss_aeNormal_dataAbnormal)
    loss_aeAbnormal_normalized = normalization(loss_aeAbnormal_dataNormal + loss_aeAbnormal_dataAbnormal)
    
    # Split back
    loss_aeNormal_dataNormal   = loss_aeNormal_normalized[:testFileNum]
    loss_aeAbnormal_dataNormal = loss_aeAbnormal_normalized[:testFileNum]
    loss_aeNormal_dataAbnormal = loss_aeNormal_normalized[testFileNum:]
    loss_aeAbnormal_dataAbnormal = loss_aeAbnormal_normalized[testFileNum:]
    
    # Plot
    scatterTestReuslt2D(
        loss_aeNormal_dataNormal,   loss_aeAbnormal_dataNormal,
        loss_aeNormal_dataAbnormal, loss_aeAbnormal_dataAbnormal
    )

    # Additional ratio-based plot
    dataNormal_levelingScores   = [
        x / y for x, y in zip(loss_aeNormal_dataNormal,   loss_aeAbnormal_dataNormal)
    ]
    dataAbnormal_levelingScores = [
        x / y for x, y in zip(loss_aeNormal_dataAbnormal, loss_aeAbnormal_dataAbnormal)
    ]
    scatterTestReuslt(dataNormal_levelingScores, dataAbnormal_levelingScores, 'double autoencoder')

def test_phase():
    """Run the test phase (from test_I.py)."""
    print("=== TEST PHASE: Starting test of normal & abnormal AEs ===")
    testSingleAEscore()
    print("=== TEST PHASE: Completed ===\n")

# ========== MAIN WITH FLAGS ========== #

def main():
    gc.collect()
    parser = argparse.ArgumentParser(description="Train/test normal & abnormal AEs.")
    parser.add_argument('-train', action='store_true', help='Run the training phase.')
    parser.add_argument('-test',  action='store_true', help='Run the testing phase.')
    args = parser.parse_args()

    if not args.train and not args.test:
        print("No flags specified. Use -train, -test, or both.")
        sys.exit(0)

    try:
        # If -train is present
        if args.train:
            train_phase()

        # If -test is present
        if args.test:
            test_phase()

    except (KeyboardInterrupt, SystemExit):
        print("Manual interruption detected.")

if __name__ == "__main__":
    main()
